#!/usr/bin/env python3
"""video_fingerprint.py の答え合わせ。  python3 -m unittest pitching/tests/test_video_fingerprint.py

fingerprint は 2 つのリポジトリ（cap-speed-analyzer と ultralytics/pitching）で
別々に計算され、突き合わせられる。しかも姿勢推定は別 PC で走るため、
**同じバイト列から必ず同じ文字列が出る**ことがこのモジュールの唯一の仕事になる。
ここがずれると、中身が同じ動画が「別の動画」と判定されて解析結果が捨てられる。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# tools/ に置いた正準実装をそのまま読む。**cap-speed-analyzer 側と同一内容。**
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from video_fingerprint import METHOD, fingerprint_bytes, fingerprint_file  # noqa: E402

MIB = 1024 * 1024


def _blob(size: int, seed: int = 0) -> bytes:
    """再現できる擬似データ。乱数種を変えれば中身が変わる。"""
    out = bytearray(size)
    value = seed & 0xFF
    for i in range(size):
        value = (value * 31 + 17) & 0xFF
        out[i] = value
    return bytes(out)


SPEC = {"duration_ms": 57359, "width": 1280, "height": 720, "fps": 239.48835806346878}


class FingerprintIsStable(unittest.TestCase):
    def test_同じ入力からは同じ値が出る(self):
        data = _blob(3 * MIB)
        self.assertEqual(fingerprint_bytes(data, **SPEC), fingerprint_bytes(data, **SPEC))

    def test_方式名が接頭辞に入る(self):
        # 方式を変えたときに、古い封筒と混ざったまま比較されないようにする。
        value = fingerprint_bytes(_blob(3 * MIB), **SPEC)
        self.assertTrue(value.startswith(METHOD + ":"), value)

    def test_ファイルから計算しても同じ値になる(self):
        data = _blob(3 * MIB)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mov"
            path.write_bytes(data)
            self.assertEqual(fingerprint_file(path, **SPEC), fingerprint_bytes(data, **SPEC))


class GoldenVectors(unittest.TestCase):
    """**別 PC の実装と突き合わせるための錨。**

    他のテストは実装を実装自身と比べているだけなので、レシピの行順を入れ替えても、
    `x` を `×` にしても、区切りを LF から , に変えても全部通ってしまう。
    そして別 PC に置いたコピーとだけ食い違う——このモジュールが防ぐはずの
    まさにその壊れ方を、テストが見逃す。ここで出力文字列そのものを固定する。

    **この値が変わったら、それはレシピを変えたということ。METHOD も変えること。**
    """

    def test_空データの既知の値(self):
        # size <= 2MiB の枝（tail が空になる経路）を踏む。
        self.assertEqual(
            fingerprint_bytes(b"", duration_ms=0, width=1, height=1, fps=1.0),
            "capfp1:8faaede5b0e543d24ff033ce84a454d2cb6e62292dc3382b3b998b9404218874")

    def test_非整数fpsの既知の値(self):
        # 実データの実測 fps。丸めが入れば必ずここで落ちる。
        self.assertEqual(
            fingerprint_bytes(b"cap", duration_ms=57359, width=1280, height=720,
                              fps=239.48835806346878),
            "capfp1:f67a66b970ca4f9e2af6a4e9a4686d4d84f53005ea3d8a2dc3319f9e7a778f78")

    def test_numpyの数値でも同じ値になる(self):
        """姿勢側は numpy 漬けで、fps が np.float64 で渡ってくることがある。

        numpy 2.x の repr(np.float64(239.5)) は 'np.float64(239.5)' であって
        '239.5' ではない。_digest の float() 強制はそれを吸収している。
        """
        class FakeNumpyFloat(float):
            def __repr__(self) -> str:  # numpy 2.x の repr を真似る
                return f"np.float64({float(self)})"

        self.assertEqual(
            fingerprint_bytes(b"cap", duration_ms=57359, width=1280, height=720,
                              fps=FakeNumpyFloat(239.48835806346878)),
            "capfp1:f67a66b970ca4f9e2af6a4e9a4686d4d84f53005ea3d8a2dc3319f9e7a778f78")


class FingerprintDiscriminates(unittest.TestCase):
    """区別できなければならないもの。"""

    def setUp(self) -> None:
        self.data = _blob(3 * MIB)
        self.base = fingerprint_bytes(self.data, **SPEC)

    def test_先頭が違えば違う値になる(self):
        changed = bytes([self.data[0] ^ 0xFF]) + self.data[1:]
        self.assertNotEqual(fingerprint_bytes(changed, **SPEC), self.base)

    def test_末尾が違えば違う値になる(self):
        # 再エンコードは末尾の moov / mdat がまず一致しない。
        changed = self.data[:-1] + bytes([self.data[-1] ^ 0xFF])
        self.assertNotEqual(fingerprint_bytes(changed, **SPEC), self.base)

    def test_真ん中だけが違う場合は区別できない(self):
        # 先頭と末尾しか読まない方式の既知の限界。黙って通すのではなく、
        # 限界であることをテストに書いて残す（サイズも長さも変わらない改変）。
        middle = len(self.data) // 2
        changed = (self.data[:middle]
                   + bytes([self.data[middle] ^ 0xFF])
                   + self.data[middle + 1:])
        self.assertEqual(fingerprint_bytes(changed, **SPEC), self.base)

    def test_サイズが違えば違う値になる(self):
        self.assertNotEqual(fingerprint_bytes(self.data + b"\x00", **SPEC), self.base)

    def test_長さが違えば違う値になる(self):
        spec = {**SPEC, "duration_ms": SPEC["duration_ms"] + 1}
        self.assertNotEqual(fingerprint_bytes(self.data, **spec), self.base)

    def test_解像度が違えば違う値になる(self):
        spec = {**SPEC, "width": 720, "height": 1280}
        self.assertNotEqual(fingerprint_bytes(self.data, **spec), self.base)

    def test_fpsが違えば違う値になる(self):
        # 240.0 と 239.488... を同じ扱いにしない。丸めない。
        spec = {**SPEC, "fps": 240.0}
        self.assertNotEqual(fingerprint_bytes(self.data, **spec), self.base)

    def test_fpsの僅差も区別する(self):
        spec = {**SPEC, "fps": 239.48835806346875}
        self.assertNotEqual(fingerprint_bytes(self.data, **spec), self.base)


class SmallFiles(unittest.TestCase):
    """2MiB 以下では先頭と末尾が重なる。同じバイトを 2 度数えない。"""

    def test_2MiB以下でも計算できる(self):
        value = fingerprint_bytes(_blob(100), **SPEC)
        self.assertTrue(value.startswith(METHOD + ":"))

    def test_2MiB以下でも中身の違いを区別する(self):
        a = fingerprint_bytes(_blob(100, seed=1), **SPEC)
        b = fingerprint_bytes(_blob(100, seed=2), **SPEC)
        self.assertNotEqual(a, b)

    def test_空ファイルでも落ちない(self):
        # 録画が失敗して 0 バイトの .mov が残ることがある。例外で止めない。
        value = fingerprint_bytes(b"", **SPEC)
        self.assertTrue(value.startswith(METHOD + ":"))

    def test_境界の前後で連続している(self):
        # 2MiB ちょうどと 2MiB+1 で別の分岐に入るが、どちらも計算できる。
        for size in (2 * MIB - 1, 2 * MIB, 2 * MIB + 1):
            with self.subTest(size=size):
                value = fingerprint_bytes(_blob(size), **SPEC)
                self.assertTrue(value.startswith(METHOD + ":"))


if __name__ == "__main__":
    unittest.main()
