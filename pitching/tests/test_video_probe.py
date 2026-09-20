#!/usr/bin/env python3
"""video_probe.py の答え合わせ。  python3 -m unittest pitching/tests/test_video_probe.py

動画ファイル自身から時間軸を読む。metadata.json の自己申告は使わない。
アプリ外で撮った動画にはそもそも metadata.json が無いため、
**ここが読めなければ何も始まらない**。

合成した最小の MOV / MP4 で検証する。実動画は要らない。
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

# tools/ に置いた正準実装をそのまま読む。**cap-speed-analyzer 側と同一内容。**
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from video_probe import (  # noqa: E402
    BROWSER_BRANDS,
    ProbeError,
    frame_to_ms,
    is_browser_playable_container,
    ms_to_frame,
    probe_video,
    spec_for_fingerprint,
)


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


def mdhd_v0(timescale: int, duration: int) -> bytes:
    # version(1) flags(3) created(4) modified(4) timescale(4) duration(4) ...
    return box(b"mdhd", bytes([0]) + b"\0\0\0" + b"\0" * 8
               + struct.pack(">II", timescale, duration) + b"\0" * 4)


def mdhd_v1(timescale: int, duration: int) -> bytes:
    # version 1 は created/modified が 64bit、duration も 64bit
    return box(b"mdhd", bytes([1]) + b"\0\0\0" + b"\0" * 16
               + struct.pack(">I", timescale) + struct.pack(">Q", duration) + b"\0" * 4)


def hdlr(handler: bytes) -> bytes:
    return box(b"hdlr", b"\0" * 8 + handler + b"\0" * 12)


def stts(entries: list[tuple[int, int]]) -> bytes:
    payload = b"\0" * 4 + struct.pack(">I", len(entries))
    for count, delta in entries:
        payload += struct.pack(">II", count, delta)
    return box(b"stts", payload)


def tkhd(width: int, height: int, matrix: tuple[int, ...] = (1, 0, 0, 0, 1, 0, 0, 0, 1)) -> bytes:
    body = bytes([0]) + b"\0\0\0" + b"\0" * (4 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2)
    body += b"".join(struct.pack(">i", v * 65536 if abs(v) <= 1 else v) for v in matrix)
    body += struct.pack(">II", width << 16, height << 16)
    return box(b"tkhd", body)


def track(handler: bytes, timescale: int, duration: int, entries, width=1280, height=720,
          mdhd_builder=mdhd_v0, matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1)) -> bytes:
    stbl = box(b"stbl", stts(entries))
    minf = box(b"minf", stbl)
    mdia = box(b"mdia", mdhd_builder(timescale, duration) + hdlr(handler) + minf)
    return box(b"trak", tkhd(width, height, matrix) + mdia)


def movie(tracks: list[bytes], moov_last: bool = True, brand: bytes = b"qt  ") -> bytes:
    ftyp = box(b"ftyp", brand + b"\0" * 8)
    mdat = box(b"mdat", b"\xAB" * 512)
    moov = box(b"moov", b"".join(tracks))
    return ftyp + mdat + moov if moov_last else ftyp + moov + mdat


# 実データに合わせた値: 240fps 相当、目盛 24000、13738 サンプル
REAL = [(10803, 100), (2935, 101)]
REAL_SAMPLES = 13738
REAL_DURATION = 10803 * 100 + 2935 * 101


class ReadsVideoTrack(unittest.TestCase):
    def probe(self, data: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(data)
            return probe_video(path)

    def test_基本的な情報が取れる(self):
        info = self.probe(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
        self.assertEqual(info["timescale"], 24000)
        self.assertEqual(info["frame_count"], REAL_SAMPLES)
        self.assertEqual(info["width"], 1280)
        self.assertEqual(info["height"], 720)
        self.assertEqual(info["rotation_deg"], 0)

    def test_読み元のパスを返す(self):
        """呼ぶ側が指紋を計算し直せるようにする。

        これが無いと、封筒の照合が「確かめていない」まま黙って通ってしまう。
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
            self.assertEqual(probe_video(path)["path"], str(path))

    def test_moovが末尾にあっても読める(self):
        # iPhone の録画は mdat のあとに moov を置く。
        info = self.probe(movie([track(b"vide", 24000, REAL_DURATION, REAL)], moov_last=True))
        self.assertEqual(info["frame_count"], REAL_SAMPLES)

    def test_moovが先頭にあっても読める(self):
        info = self.probe(movie([track(b"vide", 24000, REAL_DURATION, REAL)], moov_last=False))
        self.assertEqual(info["frame_count"], REAL_SAMPLES)

    def test_音声トラックが先にあっても映像を選ぶ(self):
        """**既存の read_timing はここで音声を返していた。**

        アプリ内の動画は映像トラックだけなので表面化していなかったが、
        アプリ外で撮った動画は音声を持つ。最初に見つけた mdhd を採ると、
        音声の目盛（44100 など）とサンプル数を映像のものとして返す。
        """
        soun = track(b"soun", 44100, 44100 * 57, [(2519424, 1)])
        vide = track(b"vide", 24000, REAL_DURATION, REAL)
        info = self.probe(movie([soun, vide]))
        self.assertEqual(info["timescale"], 24000)
        self.assertEqual(info["frame_count"], REAL_SAMPLES)

    def test_version1のボックスも読める(self):
        # 任意の .mp4 は 64bit 版を使うことがある。
        info = self.probe(movie([
            track(b"vide", 24000, REAL_DURATION, REAL, mdhd_builder=mdhd_v1)]))
        self.assertEqual(info["timescale"], 24000)
        self.assertEqual(info["frame_count"], REAL_SAMPLES)

    def test_回転が読める(self):
        info = self.probe(movie([
            track(b"vide", 24000, REAL_DURATION, REAL, width=1920, height=1080,
                  matrix=(0, 1, 0, -1, 0, 0, 0, 0, 1))]))
        self.assertEqual(info["rotation_deg"], 90)
        # 表示上は縦になる。
        self.assertEqual((info["display_width"], info["display_height"]), (1080, 1920))


class ContainerBrand(unittest.TestCase):
    """コンテナの申告を読む。**コーデックとは独立した 2 つ目の壁。**

    実データの元動画は ftyp が `qt  `（QuickTime）で、Chrome / Edge / Firefox は
    ISO-BMFF しか受けない。中身が H.264 でも、この申告のせいで弾かれる。
    別 PC から戻ってきたクリップをフォルダへ組み込む時点で検出したい。
    """

    def probe(self, brand: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)], brand=brand))
            return probe_video(path)

    def test_QuickTimeのブランドを読む(self):
        self.assertEqual(self.probe(b"qt  ")["major_brand"], "qt  ")

    def test_ISOBMFFのブランドを読む(self):
        self.assertEqual(self.probe(b"isom")["major_brand"], "isom")

    def test_QuickTimeはブラウザで再生できない扱い(self):
        self.assertFalse(is_browser_playable_container(self.probe(b"qt  ")))

    def test_ISOBMFFはブラウザで再生できる扱い(self):
        for brand in (b"isom", b"mp42", b"mp41", b"avc1"):
            with self.subTest(brand=brand):
                self.assertTrue(is_browser_playable_container(self.probe(brand)))

    def test_許可するブランドが定数として公開されている(self):
        self.assertIn("isom", BROWSER_BRANDS)
        self.assertNotIn("qt  ", BROWSER_BRANDS)


class RefusesRatherThanGuess(unittest.TestCase):
    def test_映像トラックが無ければ読めないと言う(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.m4a"
            path.write_bytes(movie([track(b"soun", 44100, 44100, [(44100, 1)])]))
            with self.assertRaises(ProbeError):
                probe_video(path)

    def test_動画でないファイルは読めないと言う(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mp4"
            path.write_bytes("これは動画ではない".encode("utf-8") * 50)
            with self.assertRaises(ProbeError):
                probe_video(path)

    def test_存在しないファイル(self):
        with self.assertRaises(ProbeError):
            probe_video(Path("/存在しない/v.mov"))

    def test_空ファイル(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mov"
            path.write_bytes(b"")
            with self.assertRaises(ProbeError):
                probe_video(path)


class TickTableIsAuthoritative(unittest.TestCase):
    """時刻の正本は目盛表。平均 fps ではない。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "v.mov"
        path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
        self.info = probe_video(path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_先頭フレームは0ms(self):
        self.assertEqual(frame_to_ms(self.info, 0), 0.0)

    def test_目盛表どおりに積み上がる(self):
        # 100 目盛 = 4.1666...ms が 10803 枚続く区間。
        self.assertAlmostEqual(frame_to_ms(self.info, 1), 100 / 24000 * 1000, places=9)
        self.assertAlmostEqual(frame_to_ms(self.info, 100), 100 * 100 / 24000 * 1000, places=9)

    def test_区間をまたいでも正しい(self):
        # 10803 枚目以降は 101 目盛になる。平均 fps で割ると必ずずれる。
        boundary = 10803
        expected = (boundary * 100) / 24000 * 1000
        self.assertAlmostEqual(frame_to_ms(self.info, boundary), expected, places=9)
        self.assertAlmostEqual(
            frame_to_ms(self.info, boundary + 1), (boundary * 100 + 101) / 24000 * 1000,
            places=9)

    def test_平均fpsで割ると途中で大きくずれる(self):
        """平均を正本にしてはいけないことを、数字で残す。

        **終端では平均は必ず収束する**（定義上そうなる）ので、そこで測っても
        ずれは見えない。ずれが最大になるのは目盛が切り替わる手前。
        実データ相当の値では frame 10801 付近で約 96ms ＝ 240fps の 23 枚分。
        骨格が 23 枚ずれて重なれば、それらしく見えたまま間違っている。
        """
        worst = max(abs(frame_to_ms(self.info, f) - f / self.info["fps"] * 1000)
                    for f in range(0, self.info["frame_count"], 7))
        self.assertGreater(worst, 50.0)

        # 公称 240fps で割ると、さらに大きい。
        nominal = max(abs(frame_to_ms(self.info, f) - f / 240.0 * 1000)
                      for f in range(0, self.info["frame_count"], 7))
        self.assertGreater(nominal, worst)

    def test_終端では平均との差が小さい(self):
        """終端だけを見て「ずれていない」と判断しないための対照。"""
        last = self.info["frame_count"] - 1
        self.assertLess(
            abs(frame_to_ms(self.info, last) - last / self.info["fps"] * 1000), 1.0)

    def test_時刻からフレームへ戻せる(self):
        for frame in (0, 1, 100, 10802, 10803, 13737):
            with self.subTest(frame=frame):
                self.assertEqual(ms_to_frame(self.info, frame_to_ms(self.info, frame)), frame)

    def test_範囲外のフレームは拒む(self):
        with self.assertRaises(ProbeError):
            frame_to_ms(self.info, self.info["frame_count"])
        with self.assertRaises(ProbeError):
            frame_to_ms(self.info, -1)

    def test_目盛表はランレングスのまま持つ(self):
        # 13738 個の配列ではなく 2 個の区間として持つ。フォルダへ置ける大きさにする。
        self.assertEqual(self.info["tick_table"], [[10803, 100], [2935, 101]])


class SpecForFingerprint(unittest.TestCase):
    def test_指紋の材料を動画から作れる(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
            spec = spec_for_fingerprint(path)
        self.assertEqual(set(spec), {"duration_ms", "width", "height", "fps"})

    def test_そのまま指紋関数へ渡せる(self):
        from video_fingerprint import METHOD, fingerprint_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
            value = fingerprint_file(path, **spec_for_fingerprint(path))
        self.assertTrue(value.startswith(METHOD + ":"))

    def test_同じファイルからは何度でも同じ材料が出る(self):
        # 2 台の PC が同じ値を出せることが前提。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(movie([track(b"vide", 24000, REAL_DURATION, REAL)]))
            self.assertEqual(spec_for_fingerprint(path), spec_for_fingerprint(path))

    def test_表示サイズではなく回転適用後を使う(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mov"
            path.write_bytes(movie([
                track(b"vide", 24000, REAL_DURATION, REAL, width=1920, height=1080,
                      matrix=(0, 1, 0, -1, 0, 0, 0, 0, 1))]))
            spec = spec_for_fingerprint(path)
        self.assertEqual((spec["width"], spec["height"]), (1080, 1920))


if __name__ == "__main__":
    unittest.main()
