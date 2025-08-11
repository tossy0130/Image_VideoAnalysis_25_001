import argparse
import cv2
import csv
import numpy as np
from collections import deque
import os

# ========= 基本サイズ =========
W, H = 960, 540

# ========= 背景更新（指数移動平均）=========
ALPHA = 0.02

# ========= 検出（ゆるめ：GOGOGO出やすく）=========
# フレーム間差分 / 背景差分のしきい
TH1_DET, TH2_DET = 8, 15
# 小領域も拾う（あとでモルフォとトラッキングで安定化）
MIN_AREA_DET = 80

# ========= トラック & 連結距離 =========
PERSIST = 5
MAX_LINK_DIST = 36

# ========= 簡易スタビライズ（整数シフト）=========
DOWNSCALE, SEARCH = 4, 3  # 1/4 に縮小して ±3px のSAD最小

# ========= ROI（検出範囲：比率）=========
BLUE_X0_RATE, BLUE_X1_RATE = 0.16, 0.86
BLUE_Y0_RATE, BLUE_Y1_RATE = 0.08, 0.92
DRAW_BLUE_ROI = False

# ========= 末尾スパイク対策 / 点灯の強さ判定 =========
EDGE_MARGIN = 8                 # warpの黒縁を無効化
# MIN_GO_AREA_STRONG = 900        # 点灯に必要な面積合計（緩め）
MIN_GO_AREA_STRONG = 750

# WIN_N, WIN_M = 5, 3             # 多数決：直近Nフレーム中M回以上で点灯（3/5）
WIN_N, WIN_M = 5, 2

GO_OFF_DECAY = 2                # 未検出時の減衰

# ========= 色（白〜薄色マスク：NumPy HSV）=========
WHITE_V, WHITE_S = 180, 80

# ========= 見た目（OpenCVでの描画。コア処理ではない）=========
BANNER_PAD = 12
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_SCALE = 1.8
BANNER_THICK = 4
GUIDE_MARGIN, GUIDE_LEN, GUIDE_GAP, GUIDE_THICK = 18, 140, 18, 4
OBJ_RECT_THICK_OUT, OBJ_RECT_THICK_IN = 5, 2

# OpenCVでの描画をやめたい場合は False（矩形のみNumPyで描く簡易版に）
USE_CV_DRAW = True

# ========= NumPyユーティリティ =========
def to_gray_numpy(bgr):
    B = bgr[..., 0].astype(np.float32)
    G = bgr[..., 1].astype(np.float32)
    R = bgr[..., 2].astype(np.float32)
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    return np.clip(gray, 0, 255).astype(np.uint8)

def hist_equalize_gray_numpy(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    cdf = np.cumsum(hist)
    cdf_norm = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-6) * 255.0
    lut = np.clip(cdf_norm, 0, 255).astype(np.uint8)
    return lut[gray]

def resize_halfstep(img, sx, sy):
    return img[::sy, ::sx]

def shift_image_numpy(img, dx, dy, fill=0):
    h, w = img.shape[:2]
    out = np.full_like(img, fill)
    x0_src = max(0, -dx)
    y0_src = max(0, -dy)
    x1_src = min(w, w - dx)
    y1_src = min(h, h - dy)
    x0_dst = max(0, dx)
    y0_dst = max(0, dy)
    x1_dst = x0_dst + (x1_src - x0_src)
    y1_dst = y0_dst + (y1_src - y0_src)
    if x1_dst > x0_dst and y1_dst > y0_dst:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out

def absdiff_numpy(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

def thresh_binary_numpy(img, th):
    return (img > th).astype(np.uint8) * 255

def and_mask_numpy(a, b):
    return ((a > 0) & (b > 0)).astype(np.uint8) * 255

def morph_open_close_numpy(mask):
    """3x3 開→閉（NumPy実装）"""
    def erode(m):
        m = (m > 0).astype(np.uint8)
        h, w = m.shape
        mp = np.pad(m, 1, mode='constant', constant_values=0)
        acc = np.ones((h, w), dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc &= mp[1+dy:1+dy+h, 1+dx:1+dx+w]
        out = np.zeros_like(m, dtype=np.uint8)
        out[acc > 0] = 255
        return out

    def dilate(m):
        m = (m > 0).astype(np.uint8)
        h, w = m.shape
        mp = np.pad(m, 1, mode='constant', constant_values=0)
        acc = np.zeros((h, w), dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= mp[1+dy:1+dy+h, 1+dx:1+dx+w]
        out = np.zeros_like(m, dtype=np.uint8)
        out[acc > 0] = 255
        return out

    opened = dilate(erode(mask))      # 開
    closed = erode(dilate(opened))    # 閉
    return closed

def hsv_mask_white_numpy(bgr, v_min=WHITE_V, s_max=WHITE_S):
    b = bgr[...,0].astype(np.float32)/255.0
    g = bgr[...,1].astype(np.float32)/255.0
    r = bgr[...,2].astype(np.float32)/255.0
    maxc = np.maximum(np.maximum(r,g), b)
    minc = np.minimum(np.minimum(r,g), b)
    v = maxc * 255.0
    s = np.zeros_like(v)
    nz = maxc > 1e-6
    s[nz] = (maxc[nz] - minc[nz]) / (maxc[nz] + 1e-6) * 255.0
    mask = ((v >= v_min) & (s <= s_max)).astype(np.uint8) * 255
    return mask

def stabilize_dxdy_numpy(prev_gray, curr_gray, down=DOWNSCALE, search=SEARCH):
    sp = resize_halfstep(prev_gray, down, down).astype(np.int16)
    sc = resize_halfstep(curr_gray, down, down).astype(np.int16)
    h, w = sp.shape
    best = (0,0); best_sad = 1e18
    for dy in range(-search, search+1):
        for dx in range(-search, search+1):
            y0 = max(0, dy); y1 = min(h, h+dy)
            x0 = max(0, dx); x1 = min(w, w+dx)
            if y1<=y0 or x1<=x0: 
                continue
            sad = np.abs(sp[y0:y1, x0:x1] - sc[y0-dy:y1-dy, x0-dx:x1-dx]).sum()
            if sad < best_sad:
                best_sad, best = sad, (dx, dy)
    return best[0]*down, best[1]*down

def connected_components_numpy(mask):
    """4近傍 二段階ラベリング（NumPy）"""
    h, w = mask.shape
    labels = np.zeros((h,w), dtype=np.int32)
    parent = [0]
    next_label = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(h):
        for x in range(w):
            if mask[y,x]==0: 
                continue
            neighbors = []
            if x>0 and labels[y,x-1]>0: neighbors.append(labels[y,x-1])
            if y>0 and labels[y-1,x]>0: neighbors.append(labels[y-1,x])
            if not neighbors:
                parent.append(next_label)
                labels[y,x] = next_label
                next_label += 1
            else:
                m = min(neighbors)
                labels[y,x] = m
                for n in neighbors:
                    union(m, n)

    rep_map = {}
    new_id = 1
    for y in range(h):
        for x in range(w):
            if labels[y,x]>0:
                r = find(labels[y,x])
                if r not in rep_map:
                    rep_map[r] = new_id
                    new_id += 1
                labels[y,x] = rep_map[r]

    n_labels = new_id
    stats = np.zeros((n_labels, 5), dtype=np.int32)  # [x,y,w,h,area]
    cents = np.zeros((n_labels, 2), dtype=np.float32)
    for y in range(h):
        xs = np.where(labels[y]>0)[0]
        for x in xs:
            lid = labels[y,x]
            stats[lid, 4] += 1
            if stats[lid,4]==1:
                stats[lid,0]=x; stats[lid,1]=y; stats[lid,2]=1; stats[lid,3]=1
            else:
                x0,y0,w0,h0 = stats[lid,0],stats[lid,1],stats[lid,2],stats[lid,3]
                x0 = min(x0, x); y0 = min(y0, y)
                x1 = max(x0+w0-1, x); y1 = max(y0+h0-1, y)
                stats[lid,0]=x0; stats[lid,1]=y0; stats[lid,2]=x1-x0+1; stats[lid,3]=y1-y0+1
            cents[lid,0]+=x; cents[lid,1]+=y

    for lid in range(1, n_labels):
        a = max(1, stats[lid,4])
        cents[lid,0] /= a; cents[lid,1] /= a

    return new_id, labels, stats, cents

# ========= トラック =========
class Track:
    _next_id = 1
    def __init__(self, cx, cy, frame_idx, bbox):
        self.id = Track._next_id; Track._next_id += 1
        self.points = deque(maxlen=80); self.points.append((cx,cy,frame_idx))
        self.bbox = bbox
        self.start = frame_idx; self.last = frame_idx
        self.done = False
    def update(self, cx, cy, frame_idx, bbox):
        self.points.append((cx,cy,frame_idx))
        self.bbox = bbox; self.last = frame_idx
    def is_valid(self):  return (self.last - self.start + 1) >= PERSIST
    def dx_total(self):  return self.points[-1][0] - self.points[0][0]
    def velocity_px_per_s(self, fps):
        if len(self.points)<2: return 0.0
        x0,_,f0 = self.points[0]; x1,_,f1 = self.points[-1]
        dt = f1 - f0
        return 0.0 if dt<=0 else (x1-x0) * (fps/dt)

# ========= メイン =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="annotated_output.mp4")
    ap.add_argument("--codec", default="avc1")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--display-every", type=int, default=2)
    args = ap.parse_args()

    # 入出力（OpenCVはI/Oとしてのみ使用）
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("動画を開けません:", args.input); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, frame0 = cap.read()
    if not ret:
        print("最初のフレーム読込に失敗"); return

    frame0 = cv2.resize(frame0, (W,H))
    gray_prev = hist_equalize_gray_numpy(to_gray_numpy(frame0))
    bg = gray_prev.astype(np.float32)

    # ROI
    bx0 = int(W*BLUE_X0_RATE); bx1 = int(W*BLUE_X1_RATE)
    by0 = int(H*BLUE_Y0_RATE); by1 = int(H*BLUE_Y1_RATE)
    detect_mask = np.zeros((H,W), np.uint8); detect_mask[by0:by1, bx0:bx1] = 255

    # 縁マスク（warpの黒縁を潰す）
    edge_mask = np.zeros((H, W), np.uint8)
    edge_mask[EDGE_MARGIN:H-EDGE_MARGIN, EDGE_MARGIN:W-EDGE_MARGIN] = 255

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.codec), fps, (W,H))
    if not vw.isOpened():
        print("VideoWriterを開けません（codec/fps/サイズ確認）"); return

    tracks, finished_rows, detection_rows = [], [], []
    frame_idx = 0
    show = not args.no_preview
    disp_every = max(1, args.display_every)

    # 多数決ウィンドウ（直近 WIN_N フレームの強検出）
    win = deque(maxlen=WIN_N)
    display_go_flag = False  # 1フレ遅延表示

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        frame = cv2.resize(frame, (W,H))

        # 色＆Gray（NumPy）
        white_mask = hsv_mask_white_numpy(frame, WHITE_V, WHITE_S)
        gray = hist_equalize_gray_numpy(to_gray_numpy(frame))

        # スタビライズ（整数SAD）
        dx, dy = stabilize_dxdy_numpy(gray_prev, gray, DOWNSCALE, SEARCH)
        gray_prev_warp = shift_image_numpy(gray_prev, dx, dy, fill=0)

        # ブレが大きい時は自動でしきい増し（誤検出抑制）
        shake = abs(dx) + abs(dy)
        if shake >= 4:
            th1_use = TH1_DET + 4
            th2_use = TH2_DET + 4
            min_area_use = MIN_AREA_DET + 40
        else:
            th1_use = TH1_DET
            th2_use = TH2_DET
            min_area_use = MIN_AREA_DET

        # 差分×2（NumPy）
        m1 = thresh_binary_numpy(absdiff_numpy(gray, gray_prev_warp), th1_use)
        m2 = thresh_binary_numpy(absdiff_numpy(gray, bg.astype(np.uint8)), th2_use)
        motion = and_mask_numpy(m1, m2)
        motion = and_mask_numpy(motion, white_mask)
        motion = and_mask_numpy(motion, detect_mask)
        motion = and_mask_numpy(motion, edge_mask)

        # モルフォ（NumPy）
        motion = morph_open_close_numpy(motion)

        # ラベリング（NumPy）
        num_labels, labels, stats, cents = connected_components_numpy(motion)

        # 検出抽出（面積しきい・ゆるめ）
        detections = []
        for lid in range(1, num_labels):
            x, y, w0, h0, area = stats[lid]
            if area < min_area_use:
                continue
            cx, cy = int(cents[lid,0]), int(cents[lid,1])
            detections.append((cx,cy,x,y,w0,h0,area))

        # ログ
        go_flag_frame = (len(detections) > 0)
        for d in detections:
            cx,cy,x,y,w0,h0,area = d
            detection_rows.append([frame_idx, cx, cy, x, y, w0, h0, int(area), 1 if go_flag_frame else 0])

        # トラッキング（最近傍）
        used = set()
        for tr in tracks:
            if tr.done: continue
            px,py,_ = tr.points[-1]
            best_j, best_d = -1, 1e18
            for j,d in enumerate(detections):
                if j in used: continue
                cx,cy = d[0], d[1]
                d2 = (cx-px)*(cx-px) + (cy-py)*(cy-py)
                if d2 < best_d:
                    best_d, best_j = d2, j
            if best_j>=0 and best_d <= (MAX_LINK_DIST**2):
                cx,cy,x,y,w0,h0,_ = detections[best_j]
                tr.update(cx,cy,frame_idx,(x,y,x+w0,y+h0))
                used.add(best_j)
            else:
                tr.done = True
                if tr.is_valid() and tr.dx_total() > 0:
                    finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])

        for j,d in enumerate(detections):
            if j in used: continue
            cx,cy,x,y,w0,h0,_ = d
            tracks.append(Track(cx,cy,frame_idx,(x,y,x+w0,y+h0)))

        # 背景更新 & 前フレ更新
        bg = (1.0-ALPHA)*bg + ALPHA*gray
        gray_prev = gray

        # ===== GOGOGO（強検出＋多数決、表示は1フレ遅延）=====
        sum_area = sum(d[-1] for d in detections)
        strong = (sum_area >= MIN_GO_AREA_STRONG)
        win.append(1 if strong else 0)
        new_go_flag = (sum(win) >= WIN_M)

        # 可視化（I/OとしてOpenCV利用。禁止なら USE_CV_DRAW=False）
        vis = frame.copy()
        if USE_CV_DRAW:
            guide_color = (255,255,0)
            for i in range(3):
                y0 = GUIDE_MARGIN + i*(GUIDE_LEN + GUIDE_GAP)
                y1 = min(y0 + GUIDE_LEN, H-10)
                cv2.rectangle(vis, (8, y0), (8+GUIDE_THICK, y1), guide_color, -1)
                cv2.rectangle(vis, (W-8-GUIDE_THICK, y0), (W-8, y1), guide_color, -1)
            if DRAW_BLUE_ROI:
                cv2.rectangle(vis,(bx0,by0),(bx1,by1),(255,128,0),2)
            for d in detections:
                cx,cy,x,y,w0,h0,area = d
                cv2.rectangle(vis,(x-2,y-2),(x+w0+2,y+h0+2),(0,128,255),OBJ_RECT_THICK_OUT)
                cv2.rectangle(vis,(x,y),(x+w0,y+h0),(0,128,255),OBJ_RECT_THICK_IN)
            text = "GOGOGO" if display_go_flag else "STOP"
            banner_color = (0,0,255) if display_go_flag else (0,255,0)
            (tw, th), _ = cv2.getTextSize(text, BANNER_FONT, BANNER_SCALE, BANNER_THICK)
            cv2.rectangle(vis, (10,10), (10+tw+BANNER_PAD*2, 10+th+BANNER_PAD*2), banner_color, -1)
            cv2.putText(vis, text, (10+BANNER_PAD, 10+BANNER_PAD+th),
                        BANNER_FONT, BANNER_SCALE, (255,255,255), BANNER_THICK, cv2.LINE_AA)
        else:
            for d in detections:
                _,_,x,y,w0,h0,_ = d
                vis[y:y+2, x:x+w0] = (0,128,255)
                vis[y+h0-2:y+h0, x:x+w0] = (0,128,255)
                vis[y:y+h0, x:x+2] = (0,128,255)
                vis[y:y+h0, x+w0-2:x+w0] = (0,128,255)

        # 出力＆プレビュー（I/O）
        vw.write(vis)
        if show and (frame_idx % disp_every == 0):
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 1フレ遅延で表示状態更新（末尾スパイク対策）
        display_go_flag = new_go_flag

    cap.release(); vw.release(); cv2.destroyAllWindows()

    # 未終了トラックも暫定で
    for tr in tracks:
        if not tr.done and tr.is_valid():
            finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])

    with open("tracks.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","start_frame","end_frame","dx_total_px","avg_speed_px_per_s"]); w.writerows(finished_rows)
    with open("detections.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["frame","cx","cy","x","y","w","h","area","detected"]); w.writerows(detection_rows)

    print("Saved video:", os.path.abspath(args.output))
    print("Saved tracks:", os.path.abspath("tracks.csv"))
    print("Saved detections:", os.path.abspath("detections.csv"))

if __name__ == "__main__":
    main()
