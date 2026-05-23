
import cv2
import sys
import copy
import numpy as np

def searchPosition(templ_img, query_img, good_match_rate=0.30):
    # A-KAZE検出器の生成
    detector = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)
    templ_kp, templ_des = detector.detectAndCompute(templ_img, None)

    # 特徴量の検出と特徴量ベクトルの計算
    query_kp, query_des = detector.detectAndCompute(query_img, None)

    # マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 特徴量ベクトル同士をマッチング
    matches = bf.match(query_des, templ_des)
    
    # 特徴量をマッチング状況に応じてソート
    matches = sorted(matches, key = lambda x:x.distance)
    good = matches[:int(len(matches) * good_match_rate)]

    # 位置計算
    src_pts = np.float32([templ_kp[m.trainIdx].pt for m in good])
    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good])
    Mx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # 画像４隅の角座標を取得
    th = templ_img.shape[0]
    tw = templ_img.shape[1]
    pts = np.array([[[0,0], [0,th-1],[tw-1,th-1],[tw-1,0]]], dtype=np.float32)
    print(pts)
    img = query_img.copy()
    dst = cv2.perspectiveTransform(pts,Mx)
    print(dst[0][0])
    cv2.rectangle(img, (dst[0][0][0], dst[0][0][1]), (dst[0][1][0], dst[0][1][1]), color=(0, 0, 255), thickness=2)
    cv2.imwrite('result.jpg', img)
    return  Mx, np.int32(dst)

# 画像を読み込む。
templ_img = cv2.imread("sample1.jpg")  # 探したい物体
query_img = cv2.imread("sample2.jpg")
ret = searchPosition(templ_img, query_img)
print(ret)


