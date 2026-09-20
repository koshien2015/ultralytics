#!/usr/bin/env python3
"""焼き込み映像がブラウザで再生できる形式で書かれることの答え合わせ。

    python3 -m unittest pitching.tests.test_video_overlay_codec

`mp4v`（MPEG-4 Part 2）は **HTML5 video がどのブラウザでも対応していない**。
デスクトップの QuickTime / VLC では再生できるので、ビューアへ載せるまで
気づけない。ここで形式を固定しておく。

利用者の確認環境は Windows の Chrome / Edge と Mac の Chrome。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pitching.visualization import video_overlay

SOURCE = Path(video_overlay.__file__)


class BrowserPlayableCodec(unittest.TestCase):
    def test_mp4vを使わない(self):
        """MPEG-4 Part 2。どのブラウザも再生できない。

        文字列そのものは「以前はこれだった」という説明で出てくるので、
        **書き出しに使っていないこと**を見る。
        """
        self.assertNotIn('VideoWriter_fourcc(*"mp4v")', SOURCE.read_text(encoding="utf-8"))

    def test_H264を既定にする(self):
        self.assertEqual(video_overlay.BROWSER_FOURCC, "avc1")

    def test_拡張子はmp4(self):
        """コンテナも直す。

        実データの元動画は `ftyp` が `qt  `（QuickTime）で、Chrome / Edge /
        Firefox は ISO-BMFF しか受けない——**コーデック以前にコンテナで弾かれる**。
        """
        self.assertEqual(video_overlay.BROWSER_SUFFIX, ".mp4")


class SuffixIsEnforced(unittest.TestCase):
    """**定数を定義しただけでは効かない。**

    avc1 で書いても、拡張子が .mov のままだとコンテナの申告が食い違う。
    §14-1 の「2 つ目の壁」（コンテナで弾かれる）を自分で作り直すことになる。
    """

    class _FakeImage:
        shape = (720, 1280, 3)

    class _FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*args):
            return 0

        @staticmethod
        def VideoWriter(*args):
            class Writer:
                def isOpened(self):
                    return True
            return Writer()

    def test_mov拡張子は拒む(self):
        with self.assertRaises(video_overlay.OverlayError) as caught:
            video_overlay._open_writer(
                self._FakeCv2, Path("/tmp/overlay.mov"), self._FakeImage(), 240.0)
        self.assertIn(".mp4", str(caught.exception))

    def test_mp4拡張子なら通る(self):
        writer = video_overlay._open_writer(
            self._FakeCv2, Path("/tmp/overlay.mp4"), self._FakeImage(), 240.0)
        self.assertTrue(writer.isOpened())

    def test_大文字の拡張子も通す(self):
        writer = video_overlay._open_writer(
            self._FakeCv2, Path("/tmp/overlay.MP4"), self._FakeImage(), 240.0)
        self.assertTrue(writer.isOpened())


class FailsLoudlyWhenCodecMissing(unittest.TestCase):
    """OpenCV の avc1 対応はビルド依存。黙って別形式で書き出させない。"""

    def test_開けなければ理由を言って止まる(self):
        class FakeWriter:
            def isOpened(self):
                return False

        class FakeCv2:
            @staticmethod
            def VideoWriter_fourcc(*args):
                return 0

            @staticmethod
            def VideoWriter(*args):
                return FakeWriter()

        class FakeImage:
            shape = (720, 1280, 3)

        with self.assertRaises(video_overlay.OverlayError) as caught:
            video_overlay._open_writer(FakeCv2, Path("/tmp/out.mp4"), FakeImage(), 240.0)
        message = str(caught.exception)
        # 何が足りないのかが分かる文言であること。
        self.assertIn("avc1", message)
        self.assertIn("H.264", message)


if __name__ == "__main__":
    unittest.main()
