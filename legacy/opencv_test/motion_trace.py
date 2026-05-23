# coding: UTF-8

import time
import cv2
import numpy as np
import argparse

VIDEO_DATA = "test_5.mp4"
outputFile = "output.mp4"

parser = argparse.ArgumentParser(description='モーションテンプレートによる動体検出')
parser.add_argument('--mask_width',   type=int,   default=100)
parser.add_argument('--mask_height',  type=int,   default=100)
parser.add_argument('--mask_start_x', type=int,   default=100)
parser.add_argument('--mask_start_y', type=int,   default=100)
parser.add_argument('--duration',     type=int,   default=1)
parser.add_argument('--area_min',     type=float, default=10.0,   help='検出対象の最小面積 (px²)')
parser.add_argument('--area_max',     type=float, default=50.0, help='検出対象の最大面積 (px²)')
args = parser.parse_args()

ESC_KEY      = 0x1b
DURATION     = args.duration
MASK_HEIGHT  = args.mask_height
MASK_WIDTH   = args.mask_width
MASK_START_X = args.mask_start_x
MASK_START_Y = args.mask_start_y
AREA_MIN     = args.area_min
AREA_MAX     = args.area_max

cv2.namedWindow("motion")
outFourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoCapture(VIDEO_DATA)
W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(outputFile, outFourcc, 30.0, (W, H))

end_flag, frame_next = video.read()
height, width, channels = frame_next.shape
motion_history = np.zeros((height, width), np.float32)
frame_pre = frame_next.copy()

morph_kernel = np.ones((3, 3), np.uint8)

while end_flag:
    # マスク生成（注目領域を白で塗った 2 値画像）
    im_mask = np.zeros((height, width, 3), np.uint8)
    im_mask = cv2.rectangle(
        im_mask,
        (MASK_START_X, MASK_START_Y),
        (MASK_START_X + MASK_WIDTH, MASK_START_Y + MASK_HEIGHT),
        (255, 255, 255), -1,
    )
    im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

    # フレーム差分 → グレースケール → マスク → 2 値化
    color_diff  = cv2.absdiff(frame_next, frame_pre)
    gray_diff   = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
    gray_diff_m = cv2.bitwise_and(gray_diff, im_mask)
    _, black_diff = cv2.threshold(gray_diff_m, 30, 255, cv2.THRESH_BINARY)

    # モルフォロジー演算: オープニングで小ノイズ除去 → クロージングで穴埋め
    clean_diff = cv2.morphologyEx(black_diff, cv2.MORPH_OPEN,  morph_kernel)
    clean_diff = cv2.morphologyEx(clean_diff, cv2.MORPH_CLOSE, morph_kernel)

    # モーション履歴の更新（silhouette は 0/1 の 2 値）
    proc_time  = time.perf_counter()
    silhouette = (clean_diff > 0).astype(np.uint8)
    cv2.motempl.updateMotionHistory(silhouette, motion_history, proc_time, DURATION)

    # 経過時間に応じてモーション履歴をフェード表示
    hist_color = np.array(
        np.clip((motion_history - (proc_time - DURATION)) / DURATION, 0, 1) * 255,
        np.uint8,
    )
    hist_gray = cv2.cvtColor(hist_color, cv2.COLOR_GRAY2BGR)

    # モーション方向を計算（結果は現状未描画。必要に応じて利用可）
    mask_grad, orientation = cv2.motempl.calcMotionGradient(
        motion_history, 0.25, 0.05, apertureSize=5
    )
    angle_deg = cv2.motempl.calcGlobalOrientation(
        orientation, mask_grad, motion_history, proc_time, DURATION
    )

    dst = cv2.addWeighted(frame_next, 1, hist_gray, 0.5, 0)

    # 輪郭検出 → 面積フィルタ → 重心描画
    contours, _ = cv2.findContours(clean_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if AREA_MIN <= area <= AREA_MAX:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(dst, (cx, cy), 6, (0, 255, 0), -1)       # 重心（緑）
                cv2.drawContours(dst, [contour], -1, (255, 255, 0), 1)  # 輪郭（水色）

    cv2.imshow("motion", dst)
    out.write(cv2.resize(dst, (W, H)))

    if cv2.waitKey(20) == ESC_KEY:
        break

    frame_pre = frame_next.copy()
    end_flag, frame_next = video.read()

out.release()
cv2.destroyAllWindows()
video.release()