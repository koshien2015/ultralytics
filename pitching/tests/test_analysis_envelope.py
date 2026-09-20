#!/usr/bin/env python3
"""analysis_envelope.py の答え合わせ。  python3 -m unittest pitching/tests/test_analysis_envelope.py

封筒は「この解析結果がどのクリップの、どの座標系のものか」を言う場所。

**1 球 1 フォルダで自己完結させる方針（設計文書 §13）になったので、
封筒はもうセッションを知らない。** 親セッションが無いクリップ（アプリ外で
撮った動画）も同じ形で扱える必要がある。

GPU も実動画も要らない。合成した最小の MOV で検証する。
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# tools/ に置いた正準実装をそのまま読む。**cap-speed-analyzer 側と同一内容。**
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analysis_envelope import (  # noqa: E402
    FRAME_ORIGIN,
    SCHEMA_VERSION,
    EnvelopeProblem,
    build_envelope,
    load_envelope,
    source_time_ms,
    verify_against_clip,
)
from video_probe import probe_video  # noqa: E402


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


def clip_bytes(entries=((480, 100),), timescale=24000, width=1280, height=720) -> bytes:
    """最小のクリップ。既定は 480 フレーム（1 球ぶんの目安）。"""
    stts_payload = b"\0" * 4 + struct.pack(">I", len(entries))
    for count, delta in entries:
        stts_payload += struct.pack(">II", count, delta)
    duration = sum(c * d for c, d in entries)
    mdhd = box(b"mdhd", bytes([0]) + b"\0\0\0" + b"\0" * 8
               + struct.pack(">II", timescale, duration) + b"\0" * 4)
    hdlr = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    minf = box(b"minf", box(b"stbl", box(b"stts", stts_payload)))
    # version(1) flags(3) created(4) modified(4) track_id(4) reserved(4) duration(4)
    # reserved(8) layer(2) alternate_group(2) volume(2) reserved(2) = 4 + 36
    tkhd_body = (bytes([0]) + b"\0\0\0" + b"\0" * (4 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2)
                 + b"".join(struct.pack(">i", v * 65536) for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
                 + struct.pack(">II", width << 16, height << 16))
    trak = box(b"trak", box(b"tkhd", tkhd_body) + box(b"mdia", mdhd + hdlr + minf))
    return box(b"ftyp", b"qt  " + b"\0" * 8) + box(b"mdat", b"\xAB" * 512) + box(b"moov", trak)


PRODUCER = {"name": "extract-pose.py", "version": "1", "model": "yolo11x-pose.pt",
            "generatedAt": "2026-09-20T12:00:00+09:00"}


class EnvelopeFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self._tmp.name) / "clip.mov"
        self.clip.write_bytes(clip_bytes())
        self.info = probe_video(self.clip)

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, **overrides) -> dict:
        envelope = build_envelope(
            analysis_id="pose-yolo11x", analysis_type="pose", clip_path=self.clip,
            pitch_id="pitch-003", producer=PRODUCER, data={"frames": []})
        envelope.update(overrides)
        return envelope


class BuildEnvelope(EnvelopeFixture):
    def test_必要な項目が揃う(self):
        e = self.make()
        for key in ("schemaVersion", "analysisId", "analysisType", "clip",
                    "timebase", "coordinateSystem", "producer", "data"):
            self.assertIn(key, e)

    def test_セッションのことは知らない(self):
        # 親セッションが無いクリップも同じ形で扱える。
        self.assertNotIn("sourceVideo", self.make())
        self.assertNotIn("segment", self.make())

    def test_クリップのbasenameだけを持つ(self):
        # 解析は別 PC で走る。あちらの絶対パスはこちらで意味を持たない。
        self.assertEqual(self.make()["clip"]["file"], "clip.mov")

    def test_目盛表が正本として入る(self):
        e = self.make()
        self.assertEqual(e["timebase"]["tickTable"], [[480, 100]])
        self.assertEqual(e["timebase"]["timescale"], 24000)

    def test_フレーム原点はクリップのみ(self):
        self.assertEqual(self.make()["timebase"]["frameOrigin"], FRAME_ORIGIN)
        self.assertEqual(FRAME_ORIGIN, "clip")

    def test_指紋が入る(self):
        from video_fingerprint import METHOD
        self.assertTrue(self.make()["clip"]["fingerprint"].startswith(METHOD + ":"))

    def test_座標系はクリップの表示寸法(self):
        e = self.make()
        self.assertEqual((e["coordinateSystem"]["width"], e["coordinateSystem"]["height"]),
                         (1280, 720))

    def test_JSONへ書き出して読み戻せる(self):
        path = Path(self._tmp.name) / "pose.v1.json"
        path.write_text(json.dumps(self.make(), ensure_ascii=False), encoding="utf-8")
        self.assertEqual(load_envelope(path)["analysisId"], "pose-yolo11x")


class VerifyAgainstClip(EnvelopeFixture):
    """このフォルダのクリップに対して計算されたものか。"""

    def test_同じクリップなら問題なし(self):
        self.assertEqual(verify_against_clip(self.make(), self.info), [])

    def test_別のクリップなら拒否する(self):
        """別 PC から戻ってきたファイルを、違うフォルダへ置いてしまった場合。

        投球ごとにクリップが違うので、ここは指紋で実際に区別できる。
        """
        other = Path(self._tmp.name) / "other.mov"
        other.write_bytes(clip_bytes(entries=((512, 100),)))
        problems = verify_against_clip(self.make(), probe_video(other))
        self.assertTrue(any(p.fatal for p in problems))
        self.assertIn("指紋", problems[0].reason)

    def test_封筒の無い旧形式は拒否するが例外にはしない(self):
        legacy = {"schema_version": 1, "meta": {"fps": 60.0}, "frames": []}
        problems = verify_against_clip(legacy, self.info)
        self.assertTrue(any(p.fatal for p in problems))
        self.assertIn("封筒", problems[0].reason)

    def test_スキーマ版が新しすぎれば拒否する(self):
        problems = verify_against_clip(self.make(schemaVersion="99.0"), self.info)
        self.assertTrue(any(p.fatal for p in problems))

    def test_解像度が違えば拒否する(self):
        e = self.make()
        e["coordinateSystem"] = {**e["coordinateSystem"], "width": 640}
        self.assertTrue(any(p.fatal for p in verify_against_clip(e, self.info)))

    def test_フレーム原点が違えば拒否する(self):
        e = self.make()
        e["timebase"] = {**e["timebase"], "frameOrigin": "source"}
        problems = verify_against_clip(e, self.info)
        self.assertTrue(any(p.fatal for p in problems))

    def test_目盛表が違えば拒否する(self):
        # クリップを作り直したのに pose が古いまま、という状態を検出する。
        e = self.make()
        e["timebase"] = {**e["timebase"], "tickTable": [[480, 101]]}
        self.assertTrue(any(p.fatal for p in verify_against_clip(e, self.info)))


class FrameMapping(EnvelopeFixture):
    """時刻は目盛表から出す。平均 fps では出さない。"""

    def test_先頭は0ms(self):
        self.assertEqual(source_time_ms(self.make(), 0), 0.0)

    def test_目盛表どおりに積み上がる(self):
        self.assertAlmostEqual(source_time_ms(self.make(), 240), 240 * 100 / 24000 * 1000,
                               places=9)

    def test_区間をまたいでも正しい(self):
        clip = Path(self._tmp.name) / "vfr.mov"
        clip.write_bytes(clip_bytes(entries=((300, 100), (180, 101))))
        envelope = build_envelope(
            analysis_id="x", analysis_type="pose", clip_path=clip,
            pitch_id="p", producer=PRODUCER, data={})
        self.assertAlmostEqual(source_time_ms(envelope, 301),
                               (300 * 100 + 101) / 24000 * 1000, places=9)

    def test_範囲外のフレームは拒む(self):
        with self.assertRaises(EnvelopeProblem):
            source_time_ms(self.make(), 480)


class BrokenInputDoesNotKillGeneration(unittest.TestCase):
    """壊れた 1 ファイルでビューア全体が作れなくならないこと。"""

    def test_不正なJSONは例外ではなく問題として返る(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.v1.json"
            path.write_text("{これはJSONではない", encoding="utf-8")
            with self.assertRaises(EnvelopeProblem):
                load_envelope(path)

    def test_存在しないファイルも同じ扱い(self):
        with self.assertRaises(EnvelopeProblem):
            load_envelope(Path("/存在しない/pose.v1.json"))

    def test_配列が来ても落ちない(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.v1.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(EnvelopeProblem):
                load_envelope(path)


class SchemaVersionIsDeclared(EnvelopeFixture):
    def test_版が定数として公開されている(self):
        self.assertEqual(self.make()["schemaVersion"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
