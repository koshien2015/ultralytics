"""学習用フレーム画像の書き出しスクリプト

動画から学習用のフレーム画像を、指定した表現（生 / モーション強調 / 3ch合成）で
書き出す。ファイル名は「動画名_フレーム番号」で表現によらず同一なので、
一度作ったYOLOラベル(.txt)はどの表現の画像にもそのまま使い回せる。

表現の切り替え = このスクリプトの再実行だけで済み、ラベリングのやり直しは不要。

表現モード:
- raw:      生フレームそのまま
- enhanced: tennis.py と同じモーション強調（元画像0.4 + 3フレーム差分0.6）
- motion3ch: R=現フレームのグレー, G=差分(前→現), B=差分(現→次)
             外観と動きを混ぜずに各チャンネルへ割り当てる実験用表現

使用例:
    # 強調画像で書き出し（現行の学習表現）
    python make_dataset.py pitch1.mp4 pitch2.mp4 --mode enhanced --out datasets/frames

    # 同じ動画から3ch合成版を生成（ラベルは同じものが使える）
    python make_dataset.py pitch1.mp4 pitch2.mp4 --mode motion3ch --out datasets/frames_3ch

    # 3フレームごとに間引いて書き出し
    python make_dataset.py pitch1.mp4 --mode enhanced --out frames --stride 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

GAMMA = 1.5  # tennis.py と同じ差分強調ガンマ
_GAMMA_LUT = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]
).astype("uint8")


def three_frame_diff(prev_frame, cur_frame, next_frame):
    """3フレーム差分（現フレームの動体だけを分離）を返す。tennis.pyと同一の処理"""
    diff_prev = cv2.absdiff(cur_frame, prev_frame)
    diff_next = cv2.absdiff(next_frame, cur_frame)
    diff = cv2.bitwise_and(diff_prev, diff_next)
    return cv2.LUT(diff, _GAMMA_LUT)


def to_enhanced(prev_frame, cur_frame, next_frame):
    """モーション強調表現（tennis.py の enhanced と同一）"""
    diff = three_frame_diff(prev_frame, cur_frame, next_frame)
    return cv2.addWeighted(cur_frame, 0.4, diff, 0.6, 0)


def to_motion3ch(prev_frame, cur_frame, next_frame):
    """外観と動きをチャンネル分離した表現

    OpenCVのBGR順で [B, G, R] = [差分(現→次), 差分(前→現), 現フレームのグレー]
    """
    gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    diff_prev = cv2.LUT(
        cv2.cvtColor(cv2.absdiff(cur_frame, prev_frame), cv2.COLOR_BGR2GRAY),
        _GAMMA_LUT,
    )
    diff_next = cv2.LUT(
        cv2.cvtColor(cv2.absdiff(next_frame, cur_frame), cv2.COLOR_BGR2GRAY),
        _GAMMA_LUT,
    )
    return cv2.merge([diff_next, diff_prev, gray])


CONVERTERS = {
    "raw": lambda prev, cur, nxt: cur,
    "enhanced": to_enhanced,
    "motion3ch": to_motion3ch,
}


def iter_frame_triplets(video_path):
    """(フレーム番号, 前, 現, 次) を順に返すジェネレータ

    フレーム番号は現フレームの0始まり連番。先頭・末尾は前後フレームが
    ないためスキップする（差分系の表現と番号を揃えるためrawでも同じ）。
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise IOError(f"動画を開けません: {video_path}")

    try:
        frames = []
        frame_index = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) == 3:
                yield frame_index - 1, frames[0], frames[1], frames[2]
                frames = frames[1:]
            frame_index += 1
    finally:
        video.release()


def export_video(video_path, out_dir, mode, stride, quality):
    """1本の動画をフレーム画像に書き出し、書き出した枚数を返す"""
    convert = CONVERTERS[mode]
    stem = video_path.stem
    count = 0

    for frame_number, prev_f, cur_f, next_f in iter_frame_triplets(video_path):
        if frame_number % stride != 0:
            continue
        image = convert(prev_f, cur_f, next_f)
        out_path = out_dir / f"{stem}_{frame_number:06d}.jpg"
        ok = cv2.imwrite(
            str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise IOError(f"書き込み失敗: {out_path}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="学習用フレーム画像の書き出し")
    parser.add_argument("videos", nargs="+", type=Path, help="入力動画ファイル")
    parser.add_argument(
        "--mode", choices=sorted(CONVERTERS), default="enhanced",
        help="画像表現（デフォルト: enhanced）",
    )
    parser.add_argument("--out", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="書き出し間隔（3なら3フレームに1枚）",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG品質（小物体が潰れないよう高めを推奨）",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride は1以上を指定してください")

    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for video_path in args.videos:
        if not video_path.is_file():
            print(f"⚠ スキップ（ファイルなし）: {video_path}")
            continue
        count = export_video(video_path, args.out, args.mode, args.stride, args.quality)
        print(f"{video_path.name}: {count}枚 → {args.out}")
        total += count

    print(f"\n✅ 合計 {total}枚を {args.mode} 表現で書き出しました")
    print("ラベル(.txt)はファイル名が同じなら全表現で共通に使えます")


if __name__ == "__main__":
    main()
