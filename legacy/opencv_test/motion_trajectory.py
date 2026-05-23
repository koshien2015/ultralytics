import time
import math
import cv2
import numpy as np

# ビデオデータ
VIDEO_DATA = "test_5.mp4"
outputFile="output.mp4"
# Esc キー
ESC_KEY = 0x1b
# モーションの残存期間(sec)
DURATION = 2.0
# 全体の方向を表示するラインの長さ
LINE_LENGTH_ALL = 60
# 座標毎の方向を表示するラインの長さ
LINE_LENGTH_GRID = 20
# 座標毎の方向を計算する間隔
GRID_WIDTH = 40
# 方向を表示するラインの丸の半径
CIRCLE_RADIUS = 2

# 表示ウィンドウの初期化
cv2.namedWindow("motion")
# ビデオデータの読み込み
outFourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoCapture(VIDEO_DATA)
out = cv2.VideoWriter(outputFile, outFourcc, 30.0,
                          (1280, 720))  # 出力先のファイルを開く

# 最初のフレームの読み込み
end_flag, frame_next = video.read()
height, width, channels = frame_next.shape
motion_history = np.zeros((height, width), np.float32)
frame_pre = frame_next.copy()

while(end_flag):
    # フレーム間の差分計算
    color_diff = cv2.absdiff(frame_next, frame_pre)

    # グレースケール変換
    gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)

    # ２値化
    retval, black_diff = cv2.threshold(gray_diff, 30, 1, cv2.THRESH_BINARY)

    # プロセッサ処理時間(sec)を取得
    proc_time = time.clock()

    # モーション履歴画像の更新
    cv2.motempl.updateMotionHistory(black_diff, motion_history, proc_time, DURATION)

    # 古いモーションの表示を経過時間に応じて薄くする
    hist_color = np.array(np.clip((motion_history - (proc_time - DURATION)) / DURATION, 0, 1) * 255, np.uint8)

    # グレースケール変換
    hist_gray = cv2.cvtColor(hist_color, cv2.COLOR_GRAY2BGR)

    # モーション履歴画像の変化方向の計算
    #   ※ orientationには各座標に対して変化方向の値（deg）が格納されます
    mask, orientation = cv2.motempl.calcMotionGradient(motion_history, 0.25, 0.05, apertureSize = 5)

    # 各座標の動きを緑色の線で描画
    width_i = GRID_WIDTH
    while width_i < width:
        height_i = GRID_WIDTH
        while height_i < height:
            cv2.circle(hist_gray, \
                       (width_i, height_i), \
                       CIRCLE_RADIUS, \
                       (0, 255, 0), \
                       2, \
                       16, \
                       0)
            angle_deg = orientation[height_i - 1][width_i - 1]
            if angle_deg > 0:
                angle_rad = math.radians(angle_deg)
                cv2.line(hist_gray, \
                         (width_i, height_i), \
                         (int(width_i + math.cos(angle_rad) * LINE_LENGTH_GRID), int(height_i + math.sin(angle_rad) * LINE_LENGTH_GRID)), \
                         (0, 255, 0), \
                         2, \
                         16, \
                         0)

            height_i += GRID_WIDTH

        width_i += GRID_WIDTH


    # 全体的なモーション方向を計算
    angle_deg = cv2.motempl.calcGlobalOrientation(orientation, mask, motion_history, proc_time, DURATION)

    # 全体の動きを黄色い線で描画
    # cv2.circle(hist_gray, \
    #            (int(width / 2), int(height / 2)), \
    #            CIRCLE_RADIUS, \
    #            (0, 215, 255), \
    #            2, \
    #            16, \
    #            0)
    # angle_rad = math.radians(angle_deg)
    # cv2.line(hist_gray, \
    #          (int(width / 2), int(height / 2)), \
    #          (int(width / 2 + math.cos(angle_rad) * LINE_LENGTH_ALL), int(height / 2 + math.sin(angle_rad) * LINE_LENGTH_ALL)), \
    #          (0, 215, 255), \
    #          2, \
    #          16, \
    #          0)

    # モーション画像を表示
    cv2.imshow("motion", hist_gray)
    resizedImage = cv2.resize(hist_gray, (1280, 720))
    out.write(resizedImage)

    # Escキー押下で終了
    if cv2.waitKey(20) == ESC_KEY:
        break

    # 次のフレームの読み込み
    frame_pre = frame_next.copy()
    end_flag, frame_next = video.read()

# 終了処理
out.release()
cv2.destroyAllWindows()
video.release()