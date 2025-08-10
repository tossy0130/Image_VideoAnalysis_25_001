import argparse
import cv2
import csv
import numpy as np
from collections import deque
import os

# ===== パラメータ =====
W, H = 960, 540
ALPHA = 0.02
TH1, TH2 = 10, 18
WHITE_V, WHITE_S = 180, 80
MIN_AREA = 100
PERSIST = 5
MAX_LINK_DIST = 36

# スタビライズ（サブピクセルON/OFF）
USE_SUBPIX = True            # ← サブピクセル位相相関を使う
PHASE_WINDOW = True          # ← ハニング窓でリーク抑制
PHASE_DOWNSCALE = 1          # ← 1=フル解像度, 2=1/2 等（速さ優先なら2も可）
PHASE_MIN_RESPONSE = 0.05    # ← 応答が低いときは不信頼→SADにフォールバック

# 従来SAD探索（フォールバック用）
DOWNSCALE_SAD = 4
SEARCH = 3

# GOGOGO 安定化
REQ_CONSEC = 2
DECAY_WHEN_MISS = 1

# 青ROI（検出範囲・比率）
BLUE_X0_RATE, BLUE_X1_RATE = 0.16, 0.86
BLUE_Y0_RATE, BLUE_Y1_RATE = 0.08, 0.92
DRAW_BLUE_ROI = False

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

def stabilize_shift_phase(prev_g, curr_g):
    """位相相関でサブピクセル平行移動を推定（必要なら縮小＆窓）"""
    a = prev_g
    b = curr_g

    # 縮小（1/2 等）
    if PHASE_DOWNSCALE > 1:
        a = cv2.resize(a, (a.shape[1]//PHASE_DOWNSCALE, a.shape[0]//PHASE_DOWNSCALE))
        b = cv2.resize(b, (b.shape[1]//PHASE_DOWNSCALE, b.shape[0]//PHASE_DOWNSCALE))

    # float32 に
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # ★ 2D Hanning 窓（幅・高さが2以上の時だけ適用）
    if PHASE_WINDOW:
        h, w = a.shape
        if h > 1 and w > 1:
            window = cv2.createHanningWindow((w, h), cv2.CV_32F)  # (width, height)
            a *= window
            b *= window
        # それ未満なら窓はスキップ（小さすぎて作れないため）

    (shift_x, shift_y), response = cv2.phaseCorrelate(a, b)

    scale = PHASE_DOWNSCALE
    return float(shift_x * scale), float(shift_y * scale), float(response)

def stabilize_shift_sad(prev_g, curr_g):
    """従来の整数SAD探索（縮小して±SEARCHの範囲）"""
    small_prev = cv2.resize(prev_g, (prev_g.shape[1]//DOWNSCALE_SAD, prev_g.shape[0]//DOWNSCALE_SAD))
    small_curr = cv2.resize(curr_g, (curr_g.shape[1]//DOWNSCALE_SAD, curr_g.shape[0]//DOWNSCALE_SAD))
    a16 = small_prev.astype(np.int16); b16 = small_curr.astype(np.int16)
    h, w = a16.shape
    best=(0,0); best_sad=1e18
    for dy in range(-SEARCH, SEARCH+1):
        for dx in range(-SEARCH, SEARCH+1):
            y0=max(0,dy); y1=min(h,h+dy); x0=max(0,dx); x1=min(w,w+dx)
            if y1<=y0 or x1<=x0: continue
            sad = np.abs(a16[y0:y1,x0:x1]-b16[y0-dy:y1-dy,x0-dx:x1-dx]).sum()
            if sad<best_sad: best_sad, best = sad,(dx,dy)
    return best[0]*DOWNSCALE_SAD, best[1]*DOWNSCALE_SAD

def warp_shift(img, dx, dy):
    """dx,dy(浮動)で前フレームを平行移動（端は0埋め）"""
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)

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
        self.points.append((cx,cy,frame_idx)); self.bbox=bbox; self.last=frame_idx
    def is_valid(self):  return (self.last - self.start + 1) >= PERSIST
    def dx_total(self):  return self.points[-1][0] - self.points[0][0]
    def velocity_px_per_s(self, fps):
        if len(self.points)<2: return 0.0
        x0,_,f0 = self.points[0]; x1,_,f1 = self.points[-1]
        dt = (f1-f0)/fps
        return 0.0 if dt<=0 else (x1-x0)/dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="annotated_output.mp4")
    ap.add_argument("--codec", default="avc1")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--display-every", type=int, default=2)
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

    bx0 = int(W*BLUE_X0_RATE); bx1 = int(W*BLUE_X1_RATE)
    by0 = int(H*BLUE_Y0_RATE); by1 = int(H*BLUE_Y1_RATE)
    detect_mask = np.zeros((H,W), np.uint8); detect_mask[by0:by1, bx0:bx1] = 255

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(args.output, fourcc, fps, (W,H))
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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,WHITE_V), (180,WHITE_S,255))
        gray = clahe_gray(to_gray(frame))

        # ---- スタビライズ（サブピクセル→弱ければSAD） ----
        if USE_SUBPIX:
            dx_f, dy_f, resp = stabilize_shift_phase(gray_prev, gray)
            if resp < PHASE_MIN_RESPONSE:
                dx_i, dy_i = stabilize_shift_sad(gray_prev, gray)
                dx, dy = dx_i, dy_i
            else:
                dx, dy = dx_f, dy_f
        else:
            dx, dy = stabilize_shift_sad(gray_prev, gray)

        gray_prev_warp = warp_shift(gray_prev, dx, dy)

        # ---- 差分×2 ----
        diff1 = cv2.absdiff(gray, gray_prev_warp)
        _, m1 = cv2.threshold(diff1, TH1, 255, cv2.THRESH_BINARY)
        diff2 = cv2.absdiff(gray, bg.astype(np.uint8))
        _, m2 = cv2.threshold(diff2, TH2, 255, cv2.THRESH_BINARY)
        motion = cv2.bitwise_and(m1, m2)
        motion = cv2.bitwise_and(motion, white_mask)
        motion = cv2.bitwise_and(motion, detect_mask)
        motion = morph_open_close(motion)

        # ---- ラベリング ----
        num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(motion)
        detections = []
        for i in range(1, num_labels):
            x,y,w,h,area = stats[i]
            if area < MIN_AREA: continue
            cx, cy = cents[i]
            detections.append((int(cx),int(cy),x,y,w,h,area))

        # ★ フレーム検出ログ
        go_flag_frame = (len(detections) > 0)
        for d in detections:
            cx, cy, x, y, w, h, area = d
            detection_rows.append([frame_idx, int(cx), int(cy), x, y, w, h, int(area), 1 if go_flag_frame else 0])

        # ---- 簡易トラッキング（最近傍） ----
        used=set()
        for tr in tracks:
            if tr.done: continue
            px,py,_ = tr.points[-1]
            best_j, best_d = -1, 1e12
            for j,d in enumerate(detections):
                if j in used: continue
                cx,cy = d[0], d[1]
                d2 = (cx-px)**2 + (cy-py)**2
                if d2 < best_d: best_d, best_j = d2, j
            if best_j>=0 and best_d <= (MAX_LINK_DIST**2):
                cx,cy,x,y,w,h,_ = detections[best_j]
                tr.update(cx,cy,frame_idx,(x,y,x+w,y+h))
                used.add(best_j)
            else:
                tr.done = True
                if tr.is_valid() and tr.dx_total() > 0:
                    finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(),
                                          round(tr.velocity_px_per_s(fps),2)])

        for j,d in enumerate(detections):
            if j in used: continue
            cx,cy,x,y,w,h,_ = d
            tracks.append(Track(cx,cy,frame_idx,(x,y,x+w,y+h)))

        # ---- 更新 ----
        bg = (1.0-ALPHA)*bg + ALPHA*gray
        gray_prev = gray

        # ---- GOGOGO（検出連続） ----
        go_frames = go_frames+1 if len(detections)>0 else max(0, go_frames-DECAY_WHEN_MISS)
        go_flag = go_frames >= REQ_CONSEC

        # ---- 可視化＆出力 ----
        vis = frame.copy()
        guide_color = (255,255,0)
        for i in range(3):
            y0 = GUIDE_MARGIN + i*(GUIDE_LEN + GUIDE_GAP); y1 = min(y0+GUIDE_LEN, H-10)
            cv2.rectangle(vis,(8,y0),(8+GUIDE_THICK,y1),guide_color,-1)
            cv2.rectangle(vis,(W-8-GUIDE_THICK,y0),(W-8,y1),guide_color,-1)
        if DRAW_BLUE_ROI: cv2.rectangle(vis,(bx0,by0),(bx1,by1),(255,128,0),2)
        for d in detections:
            cx,cy,x,y,w,h,area = d
            cv2.rectangle(vis,(x-2,y-2),(x+w+2,y+h+2),(0,128,255),OBJ_RECT_THICK_OUT)
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,128,255),OBJ_RECT_THICK_IN)
        text = "GOGOGO" if go_flag else "STOP"
        banner_color = (0,0,255) if go_flag else (0,255,0)
        (tw,th),_ = cv2.getTextSize(text,BANNER_FONT,BANNER_SCALE,BANNER_THICK)
        cv2.rectangle(vis,(10,10),(10+tw+BANNER_PAD*2,10+th+BANNER_PAD*2),banner_color,-1)
        cv2.putText(vis,text,(10+BANNER_PAD,10+BANNER_PAD+th),BANNER_FONT,BANNER_SCALE,(255,255,255),BANNER_THICK,cv2.LINE_AA)

        vw.write(vis)
        if (not args.no_preview) and (frame_idx % max(1,args.display_every)==0):
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); vw.release(); cv2.destroyAllWindows()

    # 未終了トラックも暫定で吐く
    for tr in tracks:
        if not tr.done and tr.is_valid():
            finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(),
                                  round(tr.velocity_px_per_s(fps),2)])

    with open("tracks.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","start_frame","end_frame","dx_total_px","avg_speed_px_per_s"]); w.writerows(finished_rows)
    with open("detections.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["frame","cx","cy","x","y","w","h","area","detected"]); w.writerows(detection_rows)

    print("Saved video:", os.path.abspath(args.output))
    print("Saved tracks:", os.path.abspath("tracks.csv"))
    print("Saved detections:", os.path.abspath("detections.csv"))

if __name__ == "__main__":
    main()
