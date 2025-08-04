###### 背景差分　アルゴリズムで実装
### グーグルコラボでの実行時には、コメントアウト解除　
# !pip install opencv-python

import cv2
from google.colab.patches import cv2_imshow
import time

# --- Python標準関数でRGB→グレースケール変換 ---
def to_grayscale(frame):
    height = len(frame)
    width = len(frame[0])
    gray = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = frame[y][x][2], frame[y][x][1], frame[y][x][0]
            # 人間の視覚感度に合わせて輝度を計算
            gray[y][x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


# 差分マップ
def get_diff_map(gray1, gray2, threshold=40):
    h = len(gray1)
    w = len(gray1[0])
    diff_map = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if abs(gray1[y][x] - gray2[y][x]) > threshold:
                diff_map[y][x] = 1
    return diff_map
    
# クラスタ検出 + 重心算出
def extract_clusters(diff_map, min_cluster_size=80):
    h = len(diff_map)
    w = len(diff_map[0])
    visited = [[False for _ in range(w)] for _ in range(h)]
    clusters = []

    def dfs(y, x, pixels):
        stack = [(y, x)]
        visited[y][x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny][nx] and diff_map[ny][nx] == 1:
                            visited[ny][nx] = True
                            stack.append((ny, nx))

    for y in range(h):
        for x in range(w):
            if diff_map[y][x] == 1 and not visited[y][x]:
                pixels = []
                dfs(y, x, pixels)
                if len(pixels) >= min_cluster_size:
                    sum_y = sum(p[0] for p in pixels)
                    sum_x = sum(p[1] for p in pixels)
                    cy, cx = sum_y // len(pixels), sum_x // len(pixels)
                    clusters.append((cy, cx))
    return clusters
    
# x方向の平均移動量を計算
def compute_average_dx(prev_clusters, curr_clusters):
    total_dx = 0
    count = 0
    for py, px in prev_clusters:
        closest = min(curr_clusters, key=lambda c: abs(c[1] - px) + abs(c[0] - py))
        dx = closest[1] - px
        total_dx += dx
        count += 1
    return total_dx / count if count > 0 else 0
    
    # --- メイン処理 ---
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("動画が読み込めません")
        return

    first_frame = cv2.resize(first_frame, (320, 240))
    background_gray = to_grayscale(first_frame)
    prev_clusters = []
    dx_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        gray = to_grayscale(frame)
        diff_map = get_diff_map(background_gray, gray)
        curr_clusters = extract_clusters(diff_map)

        avg_dx = compute_average_dx(prev_clusters, curr_clusters) if prev_clusters else 0
        dx_history.append(avg_dx)
        if len(dx_history) > 10:
            dx_history.pop(0)

        # 平均移動量で判定
        mean_dx = sum(dx_history) / len(dx_history) if dx_history else 0
        flowing = mean_dx > 2 and len(curr_clusters) > 0  # → 平均で右へ2px以上動いていれば

        # 表示
        display_frame = frame.copy()
        color = (0, 0, 255) if flowing else (0, 255, 0)
        text = "GOGOGO" if flowing else "STOP"
        cv2.rectangle(display_frame, (10, 10), (150, 50), color, -1)
        cv2.putText(display_frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        for cy, cx in curr_clusters:
            cv2.circle(display_frame, (cx, cy), 4, (255, 255, 0), -1)

        cv2_imshow(display_frame)
        time.sleep(0.1)

        prev_clusters = curr_clusters

    cap.release()

#  実行（ファイル名はアップロードした動画に応じて変更）
# main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Eタンクアップ（容器投入なし）.MOV")
main("/content/drive/MyDrive/動画処理・解析 2025 07-08/Fタンクアップ（容器投入中）のコピー.MOV")


