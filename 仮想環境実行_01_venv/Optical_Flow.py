### オプティカルフロー
# １つ前の画像から、どこくらい動いたかを計算する

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
        print("カメラが開けませんでした（権限 or 他アプリ使用中の可能性）")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ===== 初期フレーム取得（ウォームアップ） =====
    frame_pre = None
    for _ in range(30):
        ret, f = cap.read()
        if ret:
            frame_pre = f
    if frame_pre is None:
        print("初期フレームが取得できませんでした。")
        cap.release()
        return

    gray_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)

    # HSV準備（可視化用）。型はuint8でOK
    hsv = np.zeros_like(frame_pre)
    hsv[..., 1] = 255  # 彩度Max

    # ===== ループ =====
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得失敗。終了します。")
            break

        cv2.imshow('frame', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farnebackでオプティカルフロー
        flow = cv2.calcOpticalFlowFarneback(
            gray_pre, gray, None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0      # flags
        )

        # ベクトル→極座標（角度・大きさ）
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 角度[rad]→Hue[0..179]、大きさ→Value[0..255]
        hsv[..., 0] = (np.rad2deg(ang) / 2).astype(np.uint8)      # 0..360 -> 0..180
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        color_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('flow', color_flow)

        k = cv2.waitKey(1)  # ← ココが本題（大文字K）
        if k == ord('q'):
            break

        gray_pre = gray
        frame_pre = frame

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
