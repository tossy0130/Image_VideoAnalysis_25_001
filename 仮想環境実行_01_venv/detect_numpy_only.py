import argparse
import cv2
import csv
import numpy as np
from collections import deque
import os

# ========= パラメータ =========
W, H = 960, 540
ALPHA = 0.02
TH1, TH2 = 10, 18
WHITE_V, WHITE_S = 180, 80
MIN_AREA = 100
PERSIST = 5
MAX_LINK_DIST = 36
DOWNSCALE, SEARCH = 4, 3

# GOGOGO 安定化
REQ_CONSEC = 2
DECAY_WHEN_MISS = 1

# ROI（比率）
BLUE_X0_RATE, BLUE_X1_RATE = 0.16, 0.86
BLUE_Y0_RATE, BLUE_Y1_RATE = 0.08, 0.92
DRAW_BLUE_ROI = False

# “最後だけ誤点灯”対策（縁マスク／強検出／1フレ遅延）
EDGE_MARGIN = 8
MIN_GO_AREA = 800
GO_ON_CONSEC = 3
GO_OFF_DECAY = 2

# 出力見た目
BANNER_PAD = 12
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_SCALE = 1.8
BANNER_THICK = 4
GUIDE_MARGIN, GUIDE_LEN, GUIDE_GAP, GUIDE_THICK = 18, 140, 18, 4
OBJ_RECT_THICK_OUT, OBJ_RECT_THICK_IN = 5, 2

# OpenCVでの描画を完全に止めたい場合は False に
USE_CV_DRAW = True  # False にすると NumPyで簡易矩形のみ（テキスト無し）

# ========= NumPyユーティリティ =========
def to_gray_numpy(bgr):
    """BGR(0..255) -> Gray (0..255)  ※OpenCV不使用"""
    # B,G,R のチャンネル順（cv2.VideoCapture は BGR）
    B = bgr[..., 0].astype(np.float32)
    G = bgr[..., 1].astype(np.float32)
    R = bgr[..., 2].astype(np.float32)
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    return np.clip(gray, 0, 255).astype(np.uint8)

def hist_equalize_gray_numpy(gray):
    """簡易ヒストグラム平坦化（CLAHEの代替・グローバルEQ）"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    cdf = np.cumsum(hist)
    cdf_norm = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-6) * 255.0
    lut = np.clip(cdf_norm, 0, 255).astype(np.uint8)
    return lut[gray]

def resize_halfstep(img, sx, sy):
    """最近傍の簡易縮小：ステップ間引き（DOWNSCALE倍）"""
    return img[::sy, ::sx]

def shift_image_numpy(img, dx, dy, fill=0):
    """整数平行移動（ワープ）をNumPyで。はみ出しは fill で埋める"""
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
    """3x3 の 開→閉（NumPyのみ）。"""

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

    # 開(=erode→dilate) → 閉(=dilate→erode) だと計算重いので、最小限なら開のみでもOK。
    opened = dilate(erode(mask))
    closed = erode(dilate(opened))
    return closed

def hsv_mask_white_numpy(bgr, v_min=WHITE_V, s_max=WHITE_S):
    """BGR -> HSV（NumPy実装・簡易型）で白〜薄色のマスク"""
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
    """縮小 + ±search のSAD最小で整数(dx,dy)"""
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
    """4近傍の簡易ラベリング（NumPyのみ・2パスUF）"""
    h, w = mask.shape
    labels = np.zeros((h,w), dtype=np.int32)
    parent = [0]  # 0番は背景
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

    # 1パス目：割り当て＆Union
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

    # 代表に圧縮
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

    # 統計（bbox/面積/重心）
    n_labels = new_id
    stats = np.zeros((n_labels, 5), dtype=np.int32)  # [x,y,w,h,area]
    cents = np.zeros((n_labels, 2), dtype=np.float32)
    for y in range(h):
        xs = np.where(labels[y]>0)[0]
        for x in xs:
            lid = labels[y,x]
            stats[lid, 4] += 1
            # bbox更新
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

    return new_id, labels, stats, cents  # 0..n_labels-1（0は背景）

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

    cap = cv2.VideoCapture(args.input)  # ← 読み込みだけOpenCV
    if not cap.isOpened():
        print("動画を開けません:", args.input); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, frame0 = cap.read()
    if not ret:
        print("最初のフレーム読込に失敗"); return

    frame0 = cv2.resize(frame0, (W,H))  # I/O都合のサイズ統一は許容
    gray_prev = hist_equalize_gray_numpy(to_gray_numpy(frame0))
    bg = gray_prev.astype(np.float32)

    # ROI
    bx0 = int(W*BLUE_X0_RATE); bx1 = int(W*BLUE_X1_RATE)
    by0 = int(H*BLUE_Y0_RATE); by1 = int(H*BLUE_Y1_RATE)
    detect_mask = np.zeros((H,W), np.uint8); detect_mask[by0:by1, bx0:bx1] = 255

    # 縁マスク
    edge_mask = np.zeros((H, W), np.uint8)
    edge_mask[EDGE_MARGIN:H-EDGE_MARGIN, EDGE_MARGIN:W-EDGE_MARGIN] = 255

    # 出力（I/OとしてVideoWriterは使用）
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.codec), fps, (W,H))
    if not vw.isOpened():
        print("VideoWriterを開けません（codec/fps/サイズ確認）"); return

    tracks, finished_rows, detection_rows = [], [], []
    frame_idx = 0
    display_go_flag = False
    go_frames = 0
    show = not args.no_preview
    disp_every = max(1, args.display_every)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        frame = cv2.resize(frame, (W,H))

        # === 前処理（NumPy） ===
        white_mask = hsv_mask_white_numpy(frame, WHITE_V, WHITE_S)
        gray = hist_equalize_gray_numpy(to_gray_numpy(frame))

        # スタビライズ（整数SAD）→ 前フレームをシフト
        dx, dy = stabilize_dxdy_numpy(gray_prev, gray, DOWNSCALE, SEARCH)
        gray_prev_warp = shift_image_numpy(gray_prev, dx, dy, fill=0)

        # 差分×2（NumPy）
        m1 = thresh_binary_numpy(absdiff_numpy(gray, gray_prev_warp), TH1)
        m2 = thresh_binary_numpy(absdiff_numpy(gray, bg.astype(np.uint8)), TH2)
        motion = and_mask_numpy(m1, m2)
        motion = and_mask_numpy(motion, white_mask)
        motion = and_mask_numpy(motion, detect_mask)
        motion = and_mask_numpy(motion, edge_mask)

        # モルフォ（NumPy）
        motion = morph_open_close_numpy(motion)

        # ラベリング（NumPy）
        num_labels, labels, stats, cents = connected_components_numpy(motion)

        # 検出抽出
        detections = []
        for lid in range(1, num_labels):
            x, y, w0, h0, area = stats[lid]
            if area < MIN_AREA: 
                continue
            cx, cy = int(cents[lid,0]), int(cents[lid,1])
            detections.append((cx,cy,x,y,w0,h0,area))

        # ログ（フレーム単位）
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

        # 背景更新
        bg = (1.0-ALPHA)*bg + ALPHA*gray
        gray_prev = gray

        # GOGOGO（“強検出”の連続で点灯）※1フレ遅延表示
        sum_area = sum(d[-1] for d in detections)
        strong = (sum_area >= MIN_GO_AREA)
        if strong:
            go_frames += 1
        else:
            go_frames = max(0, go_frames - GO_OFF_DECAY)
        new_go_flag = (go_frames >= GO_ON_CONSEC)

        # 可視化（I/OとしてOpenCV利用。禁止したい場合はUSE_CV_DRAW=False）
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
            # 簡易NumPy描画（矩形のみ、テキストなし）
            for d in detections:
                _,_,x,y,w0,h0,_ = d
                vis[y:y+2, x:x+w0] = (0,128,255)   # 上辺
                vis[y+h0-2:y+h0, x:x+w0] = (0,128,255) # 下辺
                vis[y:y+h0, x:x+2] = (0,128,255)   # 左辺
                vis[y:y+h0, x+w0-2:x+w0] = (0,128,255) # 右辺

        # 書き出し＆プレビュー（I/O）
        vw.write(vis)
        if show and (frame_idx % disp_every == 0):
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 1フレ遅延で表示状態更新（終端スパイク対策）
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
