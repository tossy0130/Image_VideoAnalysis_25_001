import cv2
import sys

cap = cv2.VideoCapture(0)  # 内蔵カメラを取得

if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()



cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    cv2.imshow('Camera', frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()