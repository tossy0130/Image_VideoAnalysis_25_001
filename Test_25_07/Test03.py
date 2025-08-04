######### フレーム間差分で実装 + 移動距離
### グーグルコラボで実行時は、コメントアウト解除
# !pip install opencv-python

import cv2
from google.colab.patches import cv2_imshow
import time
import csv
from datetime import datetime

# ---- 1. RGB画像をグレースケール画像に変換 ----
def to_grayscale(frame):
    return [[int(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]) for pixel in row] for row in frame]
    
# 差分マップ生成（より感度を上げて小さな変化も検出）
def get_diff_map(prev_gray, curr_gray, threshold=40):  # 30→25にさらに下げて感度向上
    h, w = len(prev_gray), len(prev_gray[0])
    diff_map = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if abs(prev_gray[y][x] - curr_gray[y][x]) > threshold:
                diff_map[y][x] = 1
    return diff_map
    
ROI_LEFT   = 25    # 左
ROI_TOP    = 15    # 上
ROI_RIGHT  = 295   # 右
ROI_BOTTOM = 225   # 下

# --- クラスタ抽出（ROIチェックも）---
def extract_clusters(diff_map, min_cluster_size=30):
    h, w = len(diff_map), len(diff_map[0])/Users/tossy/Documents/g_画像_動画像処理_2025/Test_25_07/Test02.py
    visited = [[False]*w for _ in range(h)]
    clusters = []
    def dfs(y, x, pixels):
        stack = [(y, x)]
        visited[y][x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and diff_map[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((ny, nx))
    for y in range(h):
        for x in range(w):
            if diff_map[y][x] and not visited[y][x]:
                pixels = []
                dfs(y, x, pixels)
                if len(pixels) >= min_cluster_size:
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    # --- ROI完全内包時のみ採用 ---
                    if ROI_LEFT <= min_x and max_x <= ROI_RIGHT and ROI_TOP <= min_y and max_y <= ROI_BOTTOM:
                        clusters.append(( (min_y, min_x, max_y, max_x), (sum(ys)//len(ys), sum(xs)//len(xs)) ))
    return clusters
    
# --- クラスタ重心のみリスト化（モーションベクトル計算用） ---
def cluster_centroids(cluster_boxes):
    return [centroid for box, centroid in cluster_boxes]
        
# バウンディングボックス取得（さらに小さな物体も検出）
def get_cluster_boxes(diff_map, min_cluster_size=40):  # 50→30に変更
    h, w = len(diff_map), len(diff_map[0])
    visited = [[False]*w for _ in range(h)]
    boxes = []

    def dfs_box(y, x, pixels):
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
            if diff_map[y][x] and not visited[y][x]:
                pixels = []
                dfs_box(y, x, pixels)
                if len(pixels) >= min_cluster_size:
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)

                    # バウンディングボックスのサイズチェック
                    width = max_x - min_x
                    height = max_y - min_y

                    # 最小サイズ要件をさらに緩和
                    if (width >= 3 and height >= 3 and  # 5→3にさらに緩和
                        ROI_LEFT <= min_x and max_x <= ROI_RIGHT and
                        ROI_TOP <= min_y and max_y <= ROI_BOTTOM):
                        boxes.append((min_x, min_y, max_x, max_y))
    return boxes
  
 # モーションベクトル計算（より厳しい距離制限）
def compute_motion_vectors(prev_clusters, curr_clusters, max_distance=40):  # 60→40に変更
    motions = []
    for py, px in prev_clusters:
        closest, min_dist = None, float('inf')
        for cy, cx in curr_clusters:
            dist = abs(px-cx)+abs(py-cy)
            if dist < min_dist and dist < max_distance:
                closest, min_dist = (cy, cx), dist
        if closest:
            dy, dx = closest[0]-py, closest[1]-px
            motions.append((dx, dy))
    return motions
    
# 動きの判定をさらに緩和（左右どちらの動きも検出）
def is_fast_moving(motions, min_movement=4):
    if not motions:
        return False
    significant = [abs(dx) for dx, dy in motions if abs(dx) >= min_movement or abs(dy) >= min_movement]
    return len(significant) >= max(1, len(motions) * 0.5)  # 40%以上が有意な動き
    
# --- メイン処理 ---
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (320, 240))
    prev_gray = to_grayscale(prev_frame)
    prev_clusters = []
    prev_centroids = []

    frame_idx = 0
    recent_motion_results = []
    log_data = []
    stability_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        gray = to_grayscale(frame)

        # カメラブレ対策
        frame_stability = calculate_frame_stability(prev_gray, gray)
        stability_history.append(frame_stability)
        if len(stability_history) > 3:
            stability_history.pop(0)
        avg_stability = sum(stability_history) / len(stability_history)
        is_camera_shake = avg_stability > 22

        diff_map = get_diff_map(prev_gray, gray, threshold=40)
        cluster_boxes = extract_clusters(diff_map, min_cluster_size=30)
        curr_centroids = cluster_centroids(cluster_boxes)

        motions = compute_motion_vectors(prev_centroids, curr_centroids)

        # 動き検知
        flowing = is_fast_moving(motions, min_movement=4)

        recent_motion_results.append(1 if flowing else 0)
        if len(recent_motion_results) > 10:
            recent_motion_results.pop(0)

        dx_list = [dx for dx, _ in motions]
        dx_list_str = ",".join(map(str, dx_list))

        # ログ記録
        if len(recent_motion_results) == 10:
            status_label = "GOGOGO" if sum(recent_motion_results) >= 5 else "STOP"
            log_data.append([frame_idx, status_label] + recent_motion_results[:] + [dx_list_str, f"{avg_stability:.2f}"])

        # ---- 可視化 ----
        display_frame = frame.copy()
        # ROI
        cv2.rectangle(display_frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (128, 128, 128), 1)
        # クラスタ四角
        for (min_y, min_x, max_y, max_x), _ in cluster_boxes:
            cv2.rectangle(display_frame, (min_x, min_y), (max_x, max_y), (0, 165, 255), 2)
        # ステータス
        color = (0, 0, 255) if flowing else (0, 255, 0)
        text = "GOGOGO" if flowing else "STOP"
        cv2.rectangle(display_frame, (10, 10), (150, 50), color, -1)
        cv2.putText(display_frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if is_camera_shake:
            cv2.putText(display_frame, "CAMERA SHAKE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2_imshow(display_frame)
        time.sleep(0.05)

        prev_gray = gray
        prev_centroids = curr_centroids
        frame_idx += 1

    cap.release()

    now = datetime.now()
    timestr = now.strftime("%H%M%S")
    csv_path = f"/content/drive/MyDrive/動画処理・解析 2025 07-08/Improved_Cluster_Detect_{timestr}.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame", "status_label"] + [f"recent_{i+1}" for i in range(10)] + ["dx_list", "stability"]
        writer.writerow(header)
        for row in log_data:
            frame = row[0]
            status = row[1]
            recent_list = row[2:12]
            dx_str = row[12] if len(row) > 12 else ""
            stability_str = row[13] if len(row) > 13 else ""
            writer.writerow([frame, status] + recent_list + [dx_str, stability_str])

    print("CSV出力:", csv_path)

# ==== 実行 ====
main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Fタンクアップ（容器投入中）.MOV")
# main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Eタンクアップ（容器投入なし）.MOV")