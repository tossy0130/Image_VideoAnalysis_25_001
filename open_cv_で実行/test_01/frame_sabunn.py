import cv2
import sys
import numpy as np

def myfunc(i):
    pass

cap = cv2.VideoCapture(0)  # 内蔵カメラを取得

if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()



cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    

cv2.namedWindow('frame')

# スライドバー　UI
cv2.createTrackbar('th',
                   'frame',
                   0,
                   255,
                   myfunc)

# 一つ前のフレームをキャプチャしておく
ret, frame_pre = cap.read()
frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)

while True:

    # 現在のフレームを取得
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Camera', frame)

    # 閾値をセット　、　スライドバーから持ってきている
    th = cv2.getTrackbarPos('th', 
                            'frame')
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    ### float に変換して、　abs で絶対値をとる
    # sub = np.abs(frame.astype(np.float) - frame_pre.astype(np.float))
    sub = np.abs(frame.astype(float) - frame_pre.astype(float))
    # 0 , 1 でにちかする
    sub[sub <= th] = 0.
    sub[sub > th] = 1.

    cv2.imshow('sub', sub)

    frame_pre = frame
   

cap.release()
cv2.destroyAllWindows()