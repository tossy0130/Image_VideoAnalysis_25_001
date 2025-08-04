######### フレーム間差分 + 移動距離 ロジックで実装

###### グーグルコラボで実行時は、コメントアウト解除
# !pip install opencv-python

import cv2
from google.colab.patches import cv2_imshow
import time
import csv
from datetime import datetime

# ---- 1. RGB画像をグレースケール画像に変換 ----
def to_grayscale(frame):
    return [[int(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]) for pixel in row] for row in frame]


# ---- 2. 前フレームと現在フレームの差分マップを生成 ----
def get_diff_map(prev_gray, curr_gray, threshold=30):
    h, w = len(prev_gray), len(prev_gray[0])
    diff_map = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            # 差分がthresholdより大きい箇所を1(動き)とする
            if abs(prev_gray[y][x] - curr_gray[y][x]) > threshold:
                diff_map[y][x] = 1
    return diff_map
    
    # ---- 3. 差分マップから「動きの塊」（クラスタ）を抽出 ----
def extract_clusters(diff_map, min_cluster_size=30):
    h, w = len(diff_map), len(diff_map[0])
    visited = [[False]*w for _ in range(h)]
    clusters = []

    ### 幅優先探索でクラスタ化 ###
    def dfs(y, x, pixels):
        stack = [(y, x)]
        visited[y][x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and diff_map[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((ny, nx))

    for y in range(h):
        for x in range(w):
            # まだ訪問していない動き部分をクラスタ化
            if diff_map[y][x] and not visited[y][x]:
                pixels = []
                dfs(y, x, pixels)
                if len(pixels) >= min_cluster_size:
                    # クラスタの重心座標を計算
                    cy = sum(p[0] for p in pixels) // len(pixels)
                    cx = sum(p[1] for p in pixels) // len(pixels)
                    clusters.append((cy, cx))
    return clusters
    
# ---- 4. 前回クラスタと今回クラスタを対応づけて動きベクトル計算 ----
def compute_motion_vectors(prev_clusters, curr_clusters, max_distance=60):
    motions = []
    for py, px in prev_clusters:
        closest, min_dist = None, float('inf')
        for cy, cx in curr_clusters:
            dist = abs(px-cx)+abs(py-cy)
            if dist < min_dist and dist < max_distance:
                closest, min_dist = (cy, cx), dist
        if closest:
            # y, xの差分（移動量）を計算
            dy, dx = closest[0]-py, closest[1]-px
            motions.append((dx, dy))
    return motions
    
# ---- 5. x方向の移動量が大きい（右方向）クラスタがあるか判定 ----
def is_fast_right_moving(motions, min_dx=5):
    # 1つでも右方向にmin_dx以上動いていれば「動きあり」
    return any(dx >= min_dx for dx, _ in motions)
    
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (320,240))
    prev_gray = to_grayscale(prev_frame)
    prev_clusters = []

    # CSV 書き込み用
    frame_idx = 0
    recent_motion_results = []
    log_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320,240))
        gray = to_grayscale(frame)

        # 差分マップ生成
        diff_map = get_diff_map(prev_gray, gray, threshold=30)
        # クラスタ抽出
        curr_clusters = extract_clusters(diff_map, min_cluster_size=30)
        # 動きベクトル計算
        motions = compute_motion_vectors(prev_clusters, curr_clusters)
        # 右方向の動き判定
        flowing = is_fast_right_moving(motions, min_dx=5)

        recent_motion_results.append(1 if flowing else 0)
        if len(recent_motion_results) > 10:
            recent_motion_results.pop(0)

        dx_list = [dx for dx, _ in motions]
        dx_list_str = ",".join(map(str, dx_list))

        if len(recent_motion_results) == 10:
            status_label = "GOGOGO" if sum(recent_motion_results) >= 6 else "STOP"
            log_data.append([frame_idx, status_label] + recent_motion_results + [dx_list_str])

        display_frame = frame.copy()

        # クラスタを追尾（四角形または円）
        for cy, cx in curr_clusters:
            # 四角形（矩形）で追尾表示
            top_left = (cx - 15, cy - 15)
            bottom_right = (cx + 15, cy + 15)
            cv2.rectangle(display_frame, top_left, bottom_right, (255, 255, 0), 2)

            # 丸（円）で表示する場合は以下のコードを使う（↑をコメントアウト）
            # cv2.circle(display_frame, (cx, cy), 15, (255, 255, 0), 2)

        # 状態表示（GOGOGO or STOP）
        color = (0,0,255) if flowing else (0,255,0)
        text = "GOGOGO" if flowing else "STOP"
        cv2.rectangle(display_frame, (10,10), (150,50), color, -1)
        cv2.putText(display_frame, text, (15,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255),2)

        cv2_imshow(display_frame)
        time.sleep(0.05)

        prev_gray = gray
        prev_clusters = curr_clusters
        frame_idx += 1

    cap.release()

    # CSVログを保存
    now = datetime.now()
    timestr = now.strftime("%H%M%S")
    csv_path = f"/content/drive/MyDrive/動画処理・解析 2025 07-08/Test_Move002_history_label_log_{timestr}.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame", "status_label"] + [f"recent_{i+1}" for i in range(10)] + ["dx_list"]
        writer.writerow(header)
        for row in log_data:
            writer.writerow(row)

    print("CSV出力:", csv_path)

# 実行部分はそのままでOK
### 動きあり動画 【動】
main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Fタンクアップ（容器投入中）のコピー.MOV")

### 動きなし動画 【静】
# main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Eタンクアップ（容器投入なし）.MOV")    