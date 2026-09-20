#!/usr/bin/env python3
"""extract-pose.py の封筒出力の答え合わせ。

    python3 -m unittest pitching.tests.test_extract_pose_envelope

GPU も実動画も要らない。推論部分（extract）は呼ばず、その戻り値を模した
辞書に対して封筒だけを被せて確かめる。姿勢推定は別 PC で走るため、
**こちらで検証できるのは封筒の正しさだけ**であり、そこは全て検証する。

1 球フォルダで自己完結させる方針（設計文書 §13）なので、
封筒はセッションを知らない。入力クリップが唯一の基準。
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def _load_script():
    """ファイル名にハイフンがあるので通常の import はできない。"""
    spec = importlib.util.spec_from_file_location("extract_pose", TOOLS / "extract-pose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract_pose = _load_script()


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


# 既定は実データに寄せた可変フレームレート。100 目盛と 101 目盛が混ざる。
# 固定 100 目盛にすると平均がちょうど 240.0 になってしまい、
# 「cv2 の公称値 240.0 と実測がずれる」という現実の状況を再現できない。
VFR = ((300, 100), (180, 101))


def clip_bytes(entries=VFR, timescale=24000, width=1280, height=720) -> bytes:
    """最小のクリップ。既定は 480 フレーム（1 球ぶんの目安）・可変フレームレート。"""
    stts_payload = b"\0" * 4 + struct.pack(">I", len(entries))
    for count, delta in entries:
        stts_payload += struct.pack(">II", count, delta)
    duration = sum(c * d for c, d in entries)
    mdhd = box(b"mdhd", bytes([0]) + b"\0\0\0" + b"\0" * 8
               + struct.pack(">II", timescale, duration) + b"\0" * 4)
    hdlr = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    minf = box(b"minf", box(b"stbl", box(b"stts", stts_payload)))
    tkhd_body = (bytes([0]) + b"\0\0\0" + b"\0" * (4 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2)
                 + b"".join(struct.pack(">i", v * 65536) for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
                 + struct.pack(">II", width << 16, height << 16))
    trak = box(b"trak", box(b"tkhd", tkhd_body) + box(b"mdia", mdhd + hdlr + minf))
    return box(b"ftyp", b"qt  " + b"\0" * 8) + box(b"mdat", b"\xAB" * 512) + box(b"moov", trak)


# 目盛表から求まる平均 fps。ちょうど 240 にはならない（実データも同様）。
CLIP_FPS = sum(c for c, _ in VFR) / (sum(c * d for c, d in VFR) / 24000)


class FakeArgs:
    def __init__(self, video: str, start: int = 0, landing: int = 0, fps: float = CLIP_FPS):
        self.video = video
        self.envelope = True
        self.pose_model = "yolo11x-pose.pt"
        self.detection_model = None
        self.start = start
        self.fps = fps
        self._landing = landing


def payload_for(args: FakeArgs) -> dict:
    return {
        "schema_version": 1,
        "meta": {
            "pitch_id": "pitch-003",
            "fps": args.fps,
            "video_path": "/別のPC/にしか/無いパス/clip.mov",
            "adapter": "ultralytics(extract-pose.py)",
            "notes": [],
            "requested_start_frame": args.start,
            "seek_landing_frame": args._landing,
            "end_frame": 479,
        },
        "frames": [{"frame_index": args._landing, "timestamp_sec": 0.0, "keypoints": {}}],
    }


class WrapInEnvelope(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self._tmp.name) / "clip.mov"
        self.clip.write_bytes(clip_bytes())

    def tearDown(self):
        self._tmp.cleanup()

    def wrap(self, **kwargs) -> dict:
        args = FakeArgs(str(self.clip), **kwargs)
        return extract_pose.wrap_in_envelope(payload_for(args), args)

    def test_封筒が付く(self):
        envelope = self.wrap()
        self.assertEqual(envelope["schemaVersion"], "2.0")
        self.assertEqual(envelope["analysisType"], "pose")

    def test_元の中身がdataへそのまま入る(self):
        # 既存 JSON を巨大な単一スキーマへ移行させない。中身は触らない。
        envelope = self.wrap()
        self.assertEqual(envelope["data"]["schema_version"], 1)
        self.assertIn("frames", envelope["data"])
        self.assertIn("meta", envelope["data"])

    def test_セッションのことは知らない(self):
        envelope = self.wrap()
        self.assertNotIn("sourceVideo", envelope)
        self.assertNotIn("segment", envelope)

    def test_クリップの指紋が入る(self):
        from video_fingerprint import fingerprint_file
        from video_probe import spec_for_fingerprint

        expected = fingerprint_file(self.clip, **spec_for_fingerprint(self.clip))
        self.assertEqual(self.wrap()["clip"]["fingerprint"], expected)

    def test_別PCの絶対パスは封筒に漏れない(self):
        envelope = self.wrap()
        self.assertEqual(envelope["clip"]["file"], "clip.mov")
        self.assertNotIn("別のPC", json.dumps(envelope["clip"], ensure_ascii=False))

    def test_目盛表が正本として入る(self):
        self.assertEqual(self.wrap()["timebase"]["tickTable"], [[300, 100], [180, 101]])

    def test_平均fpsは240ちょうどではない(self):
        # 固定フレームレートの合成データだと、このテストの前提が崩れる。
        self.assertNotAlmostEqual(CLIP_FPS, 240.0, places=3)

    def test_フレーム原点はクリップ(self):
        self.assertEqual(self.wrap()["timebase"]["frameOrigin"], "clip")

    def test_中身のfpsと動画の時間軸が食い違えば止まる(self):
        """cv2 の CAP_PROP_FPS はコンテナの公称値を返すことがある。

        表示は目盛表で行うので中身の timestamp_sec は使わないが、
        食い違ったままの値を「事実」として保存はしない。
        """
        with self.assertRaises(SystemExit) as caught:
            self.wrap(fps=240.0)
        self.assertIn("240.0", str(caught.exception))

    def test_正しい封筒はビューア側の検証を通る(self):
        from analysis_envelope import verify_against_clip
        from video_probe import probe_video

        self.assertEqual(verify_against_clip(self.wrap(), probe_video(self.clip)), [])

    def test_別のクリップのフォルダへ置いたら検出される(self):
        from analysis_envelope import verify_against_clip
        from video_probe import probe_video

        other = Path(self._tmp.name) / "other.mov"
        other.write_bytes(clip_bytes(entries=((320, 100), (192, 101))))
        problems = verify_against_clip(self.wrap(), probe_video(other))
        self.assertTrue(any(p.fatal for p in problems))

    def test_動画を読めなければ封筒を作らず止まる(self):
        # 封筒を作れないまま出力すると、結び付けられない JSON が静かに増える。
        self.clip.write_bytes(b"not a video" * 100)
        with self.assertRaises(SystemExit):
            self.wrap()


class BackwardCompatibility(unittest.TestCase):
    def test_envelopeを指定しなければ従来の形のまま(self):
        source = (TOOLS / "extract-pose.py").read_text(encoding="utf-8")
        self.assertIn("if args.envelope:", source)

    def test_セッションのmetadataはもう見ない(self):
        source = (TOOLS / "extract-pose.py").read_text(encoding="utf-8")
        self.assertNotIn("session_metadata", source)


if __name__ == "__main__":
    unittest.main()
