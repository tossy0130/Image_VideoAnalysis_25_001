import argparse
import cv2
import csv
import numpy as np
from collections import deque
import os

# ===== パラメータ（必要に応じて調整） =====
W, H = 960, 540
ALPHA = 0.02
TH1, TH2 = 10, 18
WHITE_V, WHITE_S = 180, 80
MIN_AREA = 100
PERSIST = 5
MAX_LINK_DIST = 36
DOWNSCALE, SEARCH = 4, 3

# GOGOGO の安定化（デバウンス）
REQ_CONSEC = 2
DECAY_WHEN_MISS = 1

# 青ROI（検出範囲・比率 0..1）
BLUE_X0_RATE, BLUE_X1_RATE = 0.16, 0.86
BLUE_Y0_RATE, BLUE_Y1_RATE = 0.08, 0.92
DRAW_BLUE_ROI = False  # Trueで枠を描画

# 見た目
BANNER_PAD = 12
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_SCALE = 1.8
BANNER_THICK = 4
GUIDE_MARGIN, GUIDE_LEN, GUIDE_GAP, GUIDE_THICK = 18, 140, 18, 4
OBJ_RECT_THICK_OUT, OBJ_RECT_THICK_IN = 5, 2

# ===== ユーティリティ =====
def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def clahe_gray(g):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def stabilize_dxdy(prev_small, curr_small):
    h, w = prev_small.shape
    a16 = prev_small.astype(np.int16)
    b16 = curr_small.astype(np.int16)
    best, best_sad = (0,0), 1e18
    for dy in range(-SEARCH, SEARCH+1):
        for dx in range(-SEARCH, SEARCH+1):
            y0 = max(0, dy); y1 = min(h, h+dy)
            x0 = max(0, dx); x1 = min(w, w+dx)
            if y1<=y0 or x1<=x0: continue
            sad = np.abs(a16[y0:y1, x0:x1] - b16[y0-dy:y1-dy, x0-dx:x1-dx]).sum()
            if sad < best_sad: best_sad, best = sad, (dx, dy)
    return (best[0]*DOWNSCALE, best[1]*DOWNSCALE)

def shift_image(img, dx, dy):
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0)

def morph_open_close(mask):
    k = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m

class Track:
    _next_id = 1
    def __init__(self, cx, cy, frame_idx, bbox):
        self.id = Track._next_id; Track._next_id += 1
        self.points = deque(maxlen=80)
        self.points.append((cx,cy,frame_idx))
        self.bbox = bbox
        self.start = frame_idx
        self.last  = frame_idx
        self.done  = False
    def update(self, cx, cy, frame_idx, bbox):
        self.points.append((cx,cy,frame_idx))
        self.bbox = bbox
        self.last = frame_idx
    def is_valid(self):  return (self.last - self.start + 1) >= PERSIST
    def dx_total(self):  return self.points[-1][0] - self.points[0][0]
    def velocity_px_per_s(self, fps):
        if len(self.points)<2: return 0.0
        dt = self.points[-1][2]-self.points[0][2]
        return 0.0 if dt<=0 else self.dx_total()*(fps/dt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="入力動画パス")
    ap.add_argument("--output", default="annotated_output.mp4", help="出力動画パス")
    ap.add_argument("--codec", default="avc1", help="FourCC (例: avc1, mp4v, XVID)")
    ap.add_argument("--no-preview", action="store_true", help="プレビューを表示しない")
    ap.add_argument("--display-every", type=int, default=2, help="プレビュー間引き")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("動画を開けません:", args.input); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, frame0 = cap.read()
    if not ret:
        print("最初のフレーム読込に失敗"); return

    frame0 = cv2.resize(frame0, (W,H))
    gray_prev = clahe_gray(to_gray(frame0))
    bg = gray_prev.astype(np.float32)

    # 青ROI
    bx0 = int(W*BLUE_X0_RATE); bx1 = int(W*BLUE_X1_RATE)
    by0 = int(H*BLUE_Y0_RATE); by1 = int(H*BLUE_Y1_RATE)
    detect_mask = np.zeros((H,W), np.uint8)
    detect_mask[by0:by1, bx0:bx1] = 255

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if not vw.isOpened():
        print("VideoWriterを開けません（codec/fps/サイズを確認）。"); return

    tracks, finished_rows, detection_rows = [], [], []
    frame_idx, go_frames = 0, 0
    show = not args.no_preview
    disp_every = max(1, args.display_every)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        frame = cv2.resize(frame, (W,H))

        # 白っぽさ（HSV）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,WHITE_V), (180,WHITE_S,255))

        # Gray + CLAHE
        gray = clahe_gray(to_gray(frame))

        # スタビライズ
        small_prev = cv2.resize(gray_prev, (W//DOWNSCALE, H//DOWNSCALE))
        small_curr = cv2.resize(gray,      (W//DOWNSCALE, H//DOWNSCALE))
        dx, dy = stabilize_dxdy(small_prev, small_curr)
        gray_prev_warp = shift_image(gray_prev, dx, dy)

        # 差分×2
        diff1 = cv2.absdiff(gray, gray_prev_warp)
        _, m1 = cv2.threshold(diff1, TH1, 255, cv2.THRESH_BINARY)
        diff2 = cv2.absdiff(gray, bg.astype(np.uint8))
        _, m2 = cv2.threshold(diff2, TH2, 255, cv2.THRESH_BINARY)
        motion = cv2.bitwise_and(m1, m2)

        # 青ROI × 白 × 動き
        motion = cv2.bitwise_and(motion, white_mask)
        motion = cv2.bitwise_and(motion, detect_mask)
        motion = morph_open_close(motion)

        # ラベリング
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion)
        detections = []
        for i in range(1, num_labels):
            x,y,w,h,area = stats[i]
            if area < MIN_AREA: continue
            cx, cy = centroids[i]
            detections.append((int(cx),int(cy),x,y,w,h,area))

        # ★ フレーム単位の検出ログを貯める（空CSV対策）
        go_flag_frame = (len(detections) > 0)
        for d in detections:
            cx, cy, x, y, w, h, area = d
            detection_rows.append([frame_idx, int(cx), int(cy), x, y, w, h, int(area), 1 if go_flag_frame else 0])

        # トラッキング（最近傍）
        used = set()
        for tr in tracks:
            if tr.done: continue
            px,py,_ = tr.points[-1]
            best_j, best_d = -1, 1e12
            for j,d in enumerate(detections):
                if j in used: continue
                cx,cy = d[0], d[1]
                d2 = (cx-px)**2+(cy-py)**2
                if d2 < best_d: best_d, best_j = d2, j
            if best_j>=0 and best_d <= (MAX_LINK_DIST**2):
                cx,cy,x,y,w,h,_ = detections[best_j]
                tr.update(cx,cy,frame_idx,(x,y,x+w,y+h))
                used.add(best_j)
            else:
                tr.done = True
                if tr.is_valid() and tr.dx_total() > 0:
                    finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])  # 速度は後で入れる

        for j,d in enumerate(detections):
            if j in used: continue
            cx,cy,x,y,w,h,_ = d
            tracks.append(Track(cx,cy,frame_idx,(x,y,x+w,y+h)))

        # 背景/前フレ更新
        bg = (1.0-ALPHA)*bg + ALPHA*gray
        gray_prev = gray

        # GOGOGO判定（検出の連続）
        if len(detections) > 0: go_frames += 1
        else: go_frames = max(0, go_frames - DECAY_WHEN_MISS)
        go_flag = go_frames >= REQ_CONSEC

        # 可視化
        vis = frame.copy()
        guide_color = (255,255,0)
        for i in range(3):
            y0 = GUIDE_MARGIN + i*(GUIDE_LEN + GUIDE_GAP)
            y1 = min(y0 + GUIDE_LEN, H-10)
            cv2.rectangle(vis, (8, y0), (8+GUIDE_THICK, y1), guide_color, -1)
            cv2.rectangle(vis, (W-8-GUIDE_THICK, y0), (W-8, y1), guide_color, -1)

        if DRAW_BLUE_ROI:
            cv2.rectangle(vis, (bx0,by0), (bx1,by1), (255,128,0), 2)

        for d in detections:
            cx,cy,x,y,w,h,area = d
            cv2.rectangle(vis,(x-2,y-2),(x+w+2,y+h+2),(0,128,255),OBJ_RECT_THICK_OUT)
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,128,255),OBJ_RECT_THICK_IN)

        text = "GOGOGO" if go_flag else "STOP"
        banner_color = (0,0,255) if go_flag else (0,255,0)
        (tw, th), _ = cv2.getTextSize(text, BANNER_FONT, BANNER_SCALE, BANNER_THICK)
        cv2.rectangle(vis, (10,10), (10+tw+BANNER_PAD*2, 10+th+BANNER_PAD*2), banner_color, -1)
        cv2.putText(vis, text, (10+BANNER_PAD, 10+BANNER_PAD+th),
                    BANNER_FONT, BANNER_SCALE, (255,255,255), BANNER_THICK, cv2.LINE_AA)

        vw.write(vis)

        if show and (frame_idx % disp_every == 0):
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    vw.release()
    cv2.destroyAllWindows()

    # ★ 未終了トラックも暫定で吐く（空CSV対策）
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for tr in tracks:
        if not tr.done and tr.is_valid():
            finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])

    # 速度（平均）を後書きする場合は、必要に応じて追加で計算可

    # CSV出力
    with open("tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","start_frame","end_frame","dx_total_px","avg_speed_px_per_s"])
        w.writerows(finished_rows)

    with open("detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","cx","cy","x","y","w","h","area","detected"])
        w.writerows(detection_rows)

    print("Saved video:", os.path.abspath(args.output))
    print("Saved tracks:", os.path.abspath("tracks.csv"))
    print("Saved detections:", os.path.abspath("detections.csv"))

if __name__ == "__main__":
    main()
