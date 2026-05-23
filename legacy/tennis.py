import cv2
import sys
import numpy as np
#from PoseDetector import PoseDetector

MASK_HEIGHT = 1080
MASK_WIDTH = 1920
MASK_START_X = 0
MASK_START_Y = 0
THRESHOLD = 50
EROS_KERNEL_SIZE = 10
DILATION_SIZE = 5
DETECT_AREA_UPPER = 10000
DETECT_AREA_LOWER = 0


def dilation(dilationSize, kernelSize, img):  # 膨張した画像にして返す
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5 * dilationSize + 1, 5 * dilationSize + 1), (dilationSize, dilationSize))
    dilation_img = cv2.dilate(img, kernel, element)
    return dilation_img


def detect(gray_diff, thresh_diff=THRESHOLD, dilationSize=DILATION_SIZE, kernelSize=20):  # 一定面積以上の物体を検出
    retval, black_diff = cv2.threshold(
        gray_diff, thresh_diff, 255, cv2.THRESH_BINARY)  # 2値化
    dilation_img = dilation(dilationSize, kernelSize, black_diff)  # 膨張処理
    img = dilation_img.copy()
    # 収縮
    if EROS_KERNEL_SIZE > 0:
        kernel = np.ones((EROS_KERNEL_SIZE, EROS_KERNEL_SIZE), np.uint8)
        erosion = cv2.erode(dilation_img,kernel,iterations = 1)
    else:
        erosion = dilation_img
    # 境界線検出
    contours, hierarchy = cv2.findContours(
        erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ball_pos = []

    for i in range(len(contours)):  # 重心位置を計算
        count = len(contours[i])
        area = cv2.contourArea(contours[i])  # 面積計算
        x, y = 0.0, 0.0
        for j in range(count):
            x += contours[i][j][0][0]
            y += contours[i][j][0][1]

        x /= count
        y /= count
        x = int(x)
        y = int(y)
        if int(area) > DETECT_AREA_UPPER or int(area) < DETECT_AREA_LOWER :
            break
        ball_pos.append([x, y, area])

    return ball_pos, img


def displayCircle(image, ballList, thickness):
    overlay = image.copy()
    for i in range(len(ballList)):
        x = int(ballList[i][0])
        y = int(ballList[i][1])
        area = int(ballList[i][2])
        cv2.circle(image, (x, y), 10, (0, 0, 255), thickness)
        image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
        #cv2.putText(image, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
    return image


def resizeImage(image, w=2, h=2):
    height = image.shape[0]
    width = image.shape[1]
    resizedImage = cv2.resize(image, (int(width / w), int(height / h)))
    return resizedImage


def blackToColor(bImage):
    colorImage = np.array((bImage, bImage, bImage))
    colorImage = colorImage.transpose(1, 2, 0)
    return colorImage


def run(input_video_path, output_video_path=None, masked_video_path=None, enhance_video_path=None):
    """
    動画の前処理を実行

    Args:
        input_video_path: 入力動画のパス
        output_video_path: 検出円を描画した動画の出力パス（Noneの場合は生成しない）
        masked_video_path: 差分動画の出力パス（Noneの場合は生成しない）
        enhance_video_path: 強調動画の出力パス（Noneの場合は生成しない）

    Returns:
        生成されたファイルのパスの辞書
    """
    video = cv2.VideoCapture(input_video_path)  # videoファイルを読み込む
    #inFourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outFourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)

    if not video.isOpened():  # ファイルがオープンできない場合の処理.
        print("Could not open video")
        sys.exit()

    vidw = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(vidw, vidh)

    # VideoWriterオブジェクトを作成（Noneでない場合のみ）
    out = cv2.VideoWriter(output_video_path, outFourcc, fps,
                          (int(vidw), int(vidh))) if output_video_path else None
    out2 = cv2.VideoWriter(masked_video_path, outFourcc, fps,
                          (int(vidw), int(vidh))) if masked_video_path else None
    out3 = cv2.VideoWriter(enhance_video_path, outFourcc, fps,
                          (int(vidw), int(vidh))) if enhance_video_path else None

    ok, frame = video.read()  # 最初のフレームを読み込む
    if not ok:
        print('Cannot read video file')
        sys.exit()

    frame_pre = frame.copy()
    frame_count = 1  # フレーム番号を保持する変数
    while True:
        frame_count += 1  # フレーム番号をカウントアップ
        frame_next = frame.copy()
        color_diff = cv2.absdiff(frame_next, frame_pre)  # フレーム間の差分計算
        ok, frame4 = video.read()
        if not ok:
            break
        color_diff2 = cv2.absdiff(frame4, frame_next)
        im_mask = np.zeros((int(vidh), int(vidw), 3), np.uint8)
        im_mask = cv2.rectangle(im_mask, (MASK_START_X, MASK_START_Y), (
            MASK_START_X + MASK_WIDTH, MASK_START_Y + MASK_HEIGHT), (255, 255, 255), -1)
        im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
        diff = cv2.bitwise_and(color_diff, color_diff2)
        gamma = 1.5  # 1.5〜2.5ぐらいを試す
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        diff = cv2.LUT(diff, look_up_table)
        enhanced = cv2.addWeighted(frame, 0.4, diff, 0.6, 0)
        #cv2.imshow("enhanced", enhanced)  # フレームを画面表示
        #cv2.imshow("diff", diff)
        # gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)  # グレースケール変換
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff_m = cv2.bitwise_and(gray_diff, im_mask)
        retval, black_diff = cv2.threshold(
            gray_diff_m, 30, 255, cv2.THRESH_BINARY)

        ball, dilation_img = detect(gray_diff_m)

        frame = displayCircle(frame, ball, -1)  # 丸で加工
        cImage = blackToColor(dilation_img)  # 2値化画像をカラーの配列サイズと同じにする
        #frame = PoseDetector().detect_pose(frame)
        #im1 = resizeImage(frame, 2, 2)
        #im2 = resizeImage(cImage, 2, 2)
        # im_h = cv2.hconcat([im1, im2])  # 画像を横方向に連結
        #cv2.rectangle(frame, (MASK_START_X, MASK_START_Y), (MASK_START_X + MASK_WIDTH, MASK_START_Y + MASK_HEIGHT), (255, 255, 255), -1)
        frame_pre = frame_next  # 次のフレームの読み込み
        #cv2.imshow("Tracking", frame)  # フレームを画面表示
        print(frame_count)
        if out:
            out.write(frame)
        if out2:
            out2.write(diff)
        if out3:
            out3.write(enhanced)

        # 先読みしたフレームを次の処理対象にする
        frame = frame4
    video.release()
    if out:
        out.release()
    if out2:
        out2.release()
    if out3:
        out3.release()

    print(f"\nProcessing completed!")
    result_paths = {}
    if output_video_path:
        print(f"Output saved to: {output_video_path}")
        result_paths['output'] = output_video_path
    if masked_video_path:
        print(f"Masked saved to: {masked_video_path}")
        result_paths['masked'] = masked_video_path
    if enhance_video_path:
        print(f"Enhanced saved to: {enhance_video_path}")
        result_paths['enhanced'] = enhance_video_path

    return result_paths


if __name__ == '__main__':
    import sys
    import os

    # コマンドライン引数から入力ファイルを取得
    if len(sys.argv) > 1:
        inputFile = sys.argv[1]
    else:
        inputFile = "./opencv_test/test_5.mp4"

    # 出力ファイル名の設定
    base_name = os.path.splitext(os.path.basename(inputFile))[0]
    output_dir = os.path.dirname(inputFile) if os.path.dirname(inputFile) else "."

    outputFile = os.path.join(output_dir, f"{base_name}_output.mp4")
    maskedFile = os.path.join(output_dir, f"{base_name}_masked.mp4")
    enhanceFile = os.path.join(output_dir, f"{base_name}_enhance.mp4")

    print(f"Input: {inputFile}")
    print(f"Output files will be saved to: {output_dir}")

    run(inputFile, outputFile, maskedFile, enhanceFile)
