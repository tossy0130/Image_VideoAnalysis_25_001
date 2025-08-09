
######### *********************************************
### 背景差分で、背景を更新して、カメラのブレや、背景の変化に対応する
######### *********************************************

import numpy as np
import cv2
import sys

def main():
    # カメラオープン
    if len(sys.argv) == 1:
        cap = cv2.VideoCapture(0)
    else:
        try:
            cap = cv2.VideoCapture(int(sys.argv[1]))
        except:
            cap = cv2.VideoCapture(sys.argv[1])

    if not cap.isOpened():
        print("カメラが開けませんでした（権限 or 既に使用中の可能性）")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # ← 修正

    # 背景差分（MOG2）
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # 1フレーム読み込み確認
    ret, frame = cap.read()
    if not ret:
        print("最初のフレームが取得できませんでした")
        cap.release()
        return

    # ループ
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得失敗。終了します。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        # 背景差分
        sub = fgbg.apply(gray)
        cv2.imshow('sub', sub)

        # 'q'で終了
        k = cv2.waitKey(1)  # ← 修正
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':  
    main()


