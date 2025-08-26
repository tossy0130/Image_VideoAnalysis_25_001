#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# detect_numpy_only_boost_annotated.py
#
# 目的:
#   - OpenCVは「入出力(読み書き)とプレビュー/注釈描画」に限定
#   - 画像処理のコアロジックは NumPy のみで実装
#   - GOGOGO の反応を出しやすくしつつ、末尾スパイク/手ブレ誤検出を抑制
#
# 実行例:
#   python detect_numpy_only_boost_annotated.py --input input.mp4 --output out.mp4 --no-preview
#
# 主要ステップ(概要):
#   1) スタビライズ(整数SAD)で前フレームのブレを補正
#   2) フレーム間差分 + 背景差分 (しきい小さめ) → ROI/白色マスクとAND → 開閉処理
#   3) Temporal Accumulatorで時間方向に「溜まり」を作る
#   4) 連結成分ラベリング(NumPy)で検出領域抽出 → 近傍リンクでトラッキング
#   5) 面積合計 + トラック速度 + 多数決(5中2)で GOGOGO 判定
#
# 注意:
#   - C#移植時は I/O 部分を置き換え、NumPy相当処理をC#で実装すれば移行可能
#     (absdiff/threshold/erode/dilate/UFラベリング/簡易HSV/積分等)
#   - ここでは理解のため各行/各ブロックに詳細コメントを付与
#

import argparse                     # コマンドライン引数処理(標準)
import cv2                          # 入出力と描画のみで使用
import csv                          # 検出/トラック結果のCSV出力
import numpy as np                  # コアの数値演算
from collections import deque       # 追跡点や多数決ウィンドウ用
import os                           # パス表示

# ========= 基本フレームサイズ =========
W, H = 960, 540                     # 内部処理・出力の解像度(統一のためリサイズ)

# ========= 背景更新(指数移動平均) =========
ALPHA = 0.02                        # 背景 = (1-ALPHA)*過去 + ALPHA*現在

# ========= 検出(ゆるめ)しきい =========
TH1_DET, TH2_DET = 8, 15            # フレーム間/背景差分のしきい(小さめで感度↑)
MIN_AREA_DET = 80                   # 小領域でも拾う(後段のモルフォ/追跡で安定化)

# ========= トラック & 連結距離 =========
PERSIST = 5                         # 有効トラックとみなす最小継続フレーム数
MAX_LINK_DIST = 36                  # 最近傍リンクの許容距離(ピクセル)

# ========= スタビライズ(整数SAD) =========
DOWNSCALE, SEARCH = 4, 3            # 1/4縮小で ±3px の探索 → 実座標に×4

# ========= ROI(検出範囲:比率指定) =========
BLUE_X0_RATE, BLUE_X1_RATE = 0.16, 0.86
BLUE_Y0_RATE, BLUE_Y1_RATE = 0.08, 0.92
DRAW_BLUE_ROI = False               # デバッグ表示切替

# ========= 末尾スパイク/感度強化関連 =========
EDGE_MARGIN = 8                     # ワープ黒縁を無効化するための縁カット幅
MIN_GO_AREA_STRONG = 750            # GOGOGOの面積しきい(緩め)
WIN_N, WIN_M = 5, 2                 # 多数決: 直近5フレ中2回以上で点灯
GO_OFF_DECAY = 2                    # 未検出時のカウンタ減少量

# ---- Temporal Accumulator(時間方向の溜まり) ----
TEMP_DECAY = 0.80                   # 0.8なら毎フレ20%減衰
TEMP_THR   = 60                     # Acc > 60 を有効とみなす

# ---- ブレ連動しきい(大ブレ時だけ厳しく) ----
SHAKE_THR       = 4                 # |dx|+|dy| >= 4 で大ブレ判定
SHAKE_ADD_TH    = 4                 # 大ブレ時は差分しきいを加算
SHAKE_ADD_AREA  = 40                # 大ブレ時は面積もしきいを加算

# ---- トラック基準のGO(小面積でも真に動いていればGO) ----
V_MIN = 18.0                        # 平均速度(px/s)のしきい
TRACK_GO_MIN_PERSIST = 4            # トラック最低継続フレーム(速度評価の前提)

# ========= 色(白〜薄色のマスク: 簡易HSV) =========
WHITE_V, WHITE_S = 180, 80          # Value>=180 & Saturation<=80 を白〜薄色とする

# ========= 見た目(描画: I/OなのでOpenCV可) =========
BANNER_PAD = 12
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_SCALE = 1.8
BANNER_THICK = 4
GUIDE_MARGIN, GUIDE_LEN, GUIDE_GAP, GUIDE_THICK = 18, 140, 18, 4
OBJ_RECT_THICK_OUT, OBJ_RECT_THICK_IN = 5, 2
USE_CV_DRAW = True                  # FalseでNumPy簡易枠のみ(テキスト無し)

# ========= NumPyユーティリティ =========

def to_gray_numpy(bgr):
    """BGR→グレースケール(NumPyのみ)。OpenCVのcvtColorを使わない。"""
    B = bgr[..., 0].astype(np.float32)    # B成分をfloat32へ(演算誤差抑制)
    G = bgr[..., 1].astype(np.float32)    # G成分
    R = bgr[..., 2].astype(np.float32)    # R成分
    gray = 0.114 * B + 0.587 * G + 0.299 * R  # BT.601近似
    return np.clip(gray, 0, 255).astype(np.uint8)  # 0..255へクリップしてu8に戻す

def hist_equalize_gray_numpy(gray):
    """グローバルヒストグラム平坦化(CLAHEの簡易代替)。"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)  # 濃度ヒストグラム
    cdf = np.cumsum(hist)                                              # 累積分布
    cdf_norm = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-6) * 255.0     # 0..255へ正規化
    lut = np.clip(cdf_norm, 0, 255).astype(np.uint8)                   # ルックアップテーブル
    return lut[gray]                                                   # 濃度置換

def resize_halfstep(img, sx, sy):
    """ステップ間引きで縮小(最近傍相当)。"""
    return img[::sy, ::sx]  # sy, sxステップでサンプリング

def shift_image_numpy(img, dx, dy, fill=0):
    """整数平行移動(はみ出しは fill 埋め)。"""
    h, w = img.shape[:2]                      # 入力サイズ取得
    out = np.full_like(img, fill)             # 出力をfillで初期化
    x0_src = max(0, -dx); y0_src = max(0, -dy)  # ソース矩形(左上)
    x1_src = min(w, w - dx); y1_src = min(h, h - dy)  # ソース矩形(右下)
    x0_dst = max(0, dx); y0_dst = max(0, dy)          # 出力矩形(左上)
    x1_dst = x0_dst + (x1_src - x0_src)               # 出力矩形(右下)x
    y1_dst = y0_dst + (y1_src - y0_src)               # 出力矩形(右下)y
    if x1_dst > x0_dst and y1_dst > y0_dst:           # 有効領域がある時だけコピー
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out

def absdiff_numpy(a, b):
    """絶対差分(飽和に強いよう一度int16へ)。"""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

def thresh_binary_numpy(img, th):
    """二値化: img>th を255, それ以外0。"""
    return (img > th).astype(np.uint8) * 255

def and_mask_numpy(a, b):
    """マスクAND: 両方>0 の画素だけ255。"""
    return ((a > 0) & (b > 0)).astype(np.uint8) * 255

def morph_open_close_numpy(mask):
    """3x3 開→閉(NumPy)。膨張/収縮を自前実装。"""
    def erode(m):
        m = (m > 0).astype(np.uint8)                    # 真偽へ
        h, w = m.shape                                  # サイズ
        mp = np.pad(m, 1, mode='constant', constant_values=0)  # 周囲1pxを0でパディング
        acc = np.ones((h, w), dtype=np.uint8)           # 全近傍ANDの蓄積
        for dy in (-1, 0, 1):                           # 近傍の9方向を走査
            for dx in (-1, 0, 1):
                acc &= mp[1+dy:1+dy+h, 1+dx:1+dx+w]     # すべて1(=白)なら残る
        out = np.zeros_like(m, dtype=np.uint8)
        out[acc > 0] = 255
        return out

    def dilate(m):
        m = (m > 0).astype(np.uint8)                    # 真偽へ
        h, w = m.shape                                  # サイズ
        mp = np.pad(m, 1, mode='constant', constant_values=0)  # パディング
        acc = np.zeros((h, w), dtype=np.uint8)          # 近傍ORの蓄積
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= mp[1+dy:1+dy+h, 1+dx:1+dx+w]     # どれか1なら1に
        out = np.zeros_like(m, dtype=np.uint8)
        out[acc > 0] = 255
        return out

    opened = dilate(erode(mask))                        # 開: ノイズ除去
    closed = erode(dilate(opened))                      # 閉: 穴埋め/連結
    return closed

def hsv_mask_white_numpy(bgr, v_min=WHITE_V, s_max=WHITE_S):
    """簡易HSVで白〜薄色を抽出。Hは使わず V高 & S低 を条件化。"""
    b = bgr[...,0].astype(np.float32)/255.0             # 0..1 正規化
    g = bgr[...,1].astype(np.float32)/255.0
    r = bgr[...,2].astype(np.float32)/255.0
    maxc = np.maximum(np.maximum(r,g), b)               # 明るさ V ≒ max(R,G,B)
    minc = np.minimum(np.minimum(r,g), b)               # 最小
    v = maxc * 255.0                                    # 0..255へ
    s = np.zeros_like(v)
    nz = maxc > 1e-6                                    # 0除算回避
    s[nz] = (maxc[nz] - minc[nz]) / (maxc[nz] + 1e-6) * 255.0  # 彩度S
    mask = ((v >= v_min) & (s <= s_max)).astype(np.uint8) * 255 # V高 & S低
    return mask

def stabilize_dxdy_numpy(prev_gray, curr_gray, down=DOWNSCALE, search=SEARCH):
    """SAD最小の整数(dx,dy)を1/downsclの画像で求め、実座標に拡大。"""
    sp = resize_halfstep(prev_gray, down, down).astype(np.int16)  # 前フレーム縮小
    sc = resize_halfstep(curr_gray, down, down).astype(np.int16)  # 現フレーム縮小
    h, w = sp.shape
    best = (0,0); best_sad = 1e18
    for dy in range(-search, search+1):                 # 探索範囲(±search)
        for dx in range(-search, search+1):
            y0 = max(0, dy); y1 = min(h, h+dy)          # オーバーラップ領域
            x0 = max(0, dx); x1 = min(w, w+dx)
            if y1<=y0 or x1<=x0:                        # 面積0ならスキップ
                continue
            sad = np.abs(sp[y0:y1, x0:x1] - sc[y0-dy:y1-dy, x0-dx:x1-dx]).sum()  # SAD
            if sad < best_sad:                          # 最小SADを採用
                best_sad, best = sad, (dx, dy)
    return best[0]*down, best[1]*down                   # 実座標に戻す

def connected_components_numpy(mask):
    """4近傍のラベリング(2パスUnion-Find)。OpenCV不使用。"""
    h, w = mask.shape
    labels = np.zeros((h,w), dtype=np.int32)            # 各画素のラベル(0は背景)
    parent = [0]                                        # Union-Find親配列(0は背景)
    next_label = 1                                      # 次に付与するラベル

    def find(x):                                        # UF: 根を返す(経路圧縮)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):                                     # UF: 併合
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1パス目: 仮ラベル付与 + 隣接の併合
    for y in range(h):
        for x in range(w):
            if mask[y,x]==0:                            # 背景はスキップ
                continue
            neighbors = []                              # 左と上(4近傍)を参照
            if x>0 and labels[y,x-1]>0: neighbors.append(labels[y,x-1])
            if y>0 and labels[y-1,x]>0: neighbors.append(labels[y-1,x])
            if not neighbors:                           # どちらも背景
                parent.append(next_label)               # 新しい集合を作成
                labels[y,x] = next_label                # 新ラベルを振る
                next_label += 1
            else:
                m = min(neighbors)                      # 小さい方を代表に
                labels[y,x] = m
                for n in neighbors:
                    union(m, n)                         # 集合を併合

    # 代表ラベルに再マッピング(2パス目)
    rep_map = {}
    new_id = 1
    for y in range(h):
        for x in range(w):
            if labels[y,x]>0:
                r = find(labels[y,x])                   # 根を取得
                if r not in rep_map:                    # 新しい代表なら新ID付与
                    rep_map[r] = new_id
                    new_id += 1
                labels[y,x] = rep_map[r]                # 新IDで置換

    # 統計計算(バウンディングボックス/面積/重心)
    n_labels = new_id                                   # 0..n_labels-1 (0は背景)
    stats = np.zeros((n_labels, 5), dtype=np.int32)     # [x,y,w,h,area]
    cents = np.zeros((n_labels, 2), dtype=np.float32)   # [cx,cy]累積(後で割る)
    for y in range(h):
        xs = np.where(labels[y]>0)[0]                   # この行の前景x座標
        for x in xs:
            lid = labels[y,x]
            stats[lid, 4] += 1                          # 面積++
            if stats[lid,4]==1:
                stats[lid,0]=x; stats[lid,1]=y; stats[lid,2]=1; stats[lid,3]=1  # 初期bbox
            else:
                x0,y0,w0,h0 = stats[lid,0],stats[lid,1],stats[lid,2],stats[lid,3]
                x0 = min(x0, x); y0 = min(y0, y)        # 左上更新
                x1 = max(x0+w0-1, x); y1 = max(y0+h0-1, y)  # 右下更新
                stats[lid,0]=x0; stats[lid,1]=y0; stats[lid,2]=x1-x0+1; stats[lid,3]=y1-y0+1
            cents[lid,0]+=x; cents[lid,1]+=y            # 重心用の和

    for lid in range(1, n_labels):
        a = max(1, stats[lid,4])                        # 0除算回避
        cents[lid,0] /= a; cents[lid,1] /= a            # 重心 = 総和/面積

    return new_id, labels, stats, cents                 # 0は背景、1..n_labels-1が前景

# ========= トラック(最近傍リンク) =========
class Track:
    _next_id = 1
    def __init__(self, cx, cy, frame_idx, bbox):
        self.id = Track._next_id; Track._next_id += 1   # ユニークID
        self.points = deque(maxlen=80)                  # (cx,cy,frame_idx)の履歴
        self.points.append((cx,cy,frame_idx))
        self.bbox = bbox                                # (x0,y0,x1,y1)
        self.start = frame_idx; self.last = frame_idx   # 開始/最終フレ番号
        self.done = False                               # 終了フラグ
    def update(self, cx, cy, frame_idx, bbox):
        self.points.append((cx,cy,frame_idx))           # 新しい観測点を追加
        self.bbox = bbox; self.last = frame_idx
    def is_valid(self):
        return (self.last - self.start + 1) >= PERSIST  # 最低継続フレ数に達したか
    def dx_total(self):
        return self.points[-1][0] - self.points[0][0]   # x移動量(右向き正)
    def velocity_px_per_s(self, fps):
        if len(self.points)<2: return 0.0
        x0,_,f0 = self.points[0]; x1,_,f1 = self.points[-1]  # 最初と最後
        dt = f1 - f0
        return 0.0 if dt<=0 else (x1-x0) * (fps/dt)     # 平均速度(px/s)

# ========= メイン =========
def main():
    ap = argparse.ArgumentParser()                      # 引数パーサ
    ap.add_argument("--input", required=True)           # 入力動画パス
    ap.add_argument("--output", default="annotated_output.mp4")  # 出力動画パス
    ap.add_argument("--codec", default="avc1")          # FourCC (avc1/mp4v/XVIDなど)
    ap.add_argument("--no-preview", action="store_true")# プレビュー抑制
    ap.add_argument("--display-every", type=int, default=2)  # 何フレごとに表示
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)                  # OpenCVで入力(=I/O)
    if not cap.isOpened():
        print("動画を開けません:", args.input); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0            # FPS取得(無い場合30仮定)
    ret, frame0 = cap.read()                            # 最初のフレーム
    if not ret:
        print("最初のフレーム読込に失敗"); return

    frame0 = cv2.resize(frame0, (W,H))                  # 内部サイズに正規化
    gray_prev = hist_equalize_gray_numpy(to_gray_numpy(frame0)) # Gray+EQ
    bg = gray_prev.astype(np.float32)                   # 背景(浮動小数で保持)

    # ROIマスクの作成(青枠領域のみ=255)
    bx0 = int(W*BLUE_X0_RATE); bx1 = int(W*BLUE_X1_RATE)
    by0 = int(H*BLUE_Y0_RATE); by1 = int(H*BLUE_Y1_RATE)
    detect_mask = np.zeros((H,W), np.uint8); detect_mask[by0:by1, bx0:bx1] = 255

    # 縁マスク(ワープの黒縁を無効化)
    edge_mask = np.zeros((H, W), np.uint8)
    edge_mask[EDGE_MARGIN:H-EDGE_MARGIN, EDGE_MARGIN:W-EDGE_MARGIN] = 255

    # 出力Writer(=I/O)
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.codec), fps, (W,H))
    if not vw.isOpened():
        print("VideoWriterを開けません（codec/fps/サイズ確認）"); return

    tracks, finished_rows, detection_rows = [], [], []  # ログ用配列
    frame_idx = 0                                       # 現在フレ番号
    show = not args.no_preview                          # プレビュー有無
    disp_every = max(1, args.display_every)             # 表示間引き

    win = deque(maxlen=WIN_N)                           # 多数決ウィンドウ
    display_go_flag = False                             # 1フレ遅延表示の現状態

    motion_acc = np.zeros((H, W), np.float32)           # Temporal Accumulator

    # ===== フレームループ =====
    while True:
        ret, frame = cap.read()                         # 1フレ取得
        if not ret: break                               # EOFなら終了
        frame_idx += 1                                  # フレ番号++
        frame = cv2.resize(frame, (W,H))                # 内部サイズに調整

        white_mask = hsv_mask_white_numpy(frame, WHITE_V, WHITE_S)     # 白〜薄色マスク
        gray = hist_equalize_gray_numpy(to_gray_numpy(frame))          # Gray+EQ

        dx, dy = stabilize_dxdy_numpy(gray_prev, gray, DOWNSCALE, SEARCH)  # 整列dx,dy
        gray_prev_warp = shift_image_numpy(gray_prev, dx, dy, fill=0)      # 前フレを平行移動

        shake = abs(dx) + abs(dy)                          # ブレ量(マンハッタン距離)
        if shake >= SHAKE_THR:                             # 大ブレ時はしきい増し
            th1_use = TH1_DET + SHAKE_ADD_TH
            th2_use = TH2_DET + SHAKE_ADD_TH
            min_area_use = MIN_AREA_DET + SHAKE_ADD_AREA
        else:                                              # 通常時は基準値
            th1_use = TH1_DET
            th2_use = TH2_DET
            min_area_use = MIN_AREA_DET

        m1 = thresh_binary_numpy(absdiff_numpy(gray, gray_prev_warp), th1_use)  # フレ間差分
        m2 = thresh_binary_numpy(absdiff_numpy(gray, bg.astype(np.uint8)), th2_use)  # 背景差分
        motion = and_mask_numpy(m1, m2)                     # ANDで共通部分のみ
        motion = and_mask_numpy(motion, white_mask)         # 白〜薄色制限
        motion = and_mask_numpy(motion, detect_mask)        # ROI制限
        motion = and_mask_numpy(motion, edge_mask)          # 縁を無効化

        motion = morph_open_close_numpy(motion)             # 開→閉でノイズ除去/連結

        # ---- Temporal Accumulator: 連続する弱い動きを救済 ----
        motion_acc = np.maximum(motion_acc * TEMP_DECAY, motion.astype(np.float32))
        motion_temporal = (motion_acc > TEMP_THR).astype(np.uint8) * 255

        # ラベリング(NumPy)
        num_labels, labels, stats, cents = connected_components_numpy(motion_temporal)

        # 検出抽出(面積しきい)
        detections = []
        for lid in range(1, num_labels):                    # 0は背景
            x, y, w0, h0, area = stats[lid]
            if area < min_area_use:                         # 小さすぎる領域を除外
                continue
            cx, cy = int(cents[lid,0]), int(cents[lid,1])   # 重心
            detections.append((cx,cy,x,y,w0,h0,area))       # 検出を保存

        # 検出ログを保存
        go_flag_frame = (len(detections) > 0)
        for d in detections:
            cx,cy,x,y,w0,h0,area = d
            detection_rows.append([frame_idx, cx, cy, x, y, w0, h0, int(area), 1 if go_flag_frame else 0])

        # 近傍リンクでトラッキング
        used = set()
        for tr in tracks:
            if tr.done: continue
            px,py,_ = tr.points[-1]                         # 直近の位置
            best_j, best_d = -1, 1e18
            for j,d in enumerate(detections):
                if j in used: continue
                cx,cy = d[0], d[1]
                d2 = (cx-px)*(cx-px) + (cy-py)*(cy-py)      # 2乗距離
                if d2 < best_d:
                    best_d, best_j = d2, j
            if best_j>=0 and best_d <= (MAX_LINK_DIST**2):  # 閾値以内なら結合
                cx,cy,x,y,w0,h0,_ = detections[best_j]
                tr.update(cx,cy,frame_idx,(x,y,x+w0,y+h0))  # 更新
                used.add(best_j)
            else:
                tr.done = True                               # 見失い→終了候補
                if tr.is_valid() and tr.dx_total() > 0:      # 有効かつ右向き移動なら記録
                    finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])

        # 新規トラックを生成
        for j,d in enumerate(detections):
            if j in used: continue
            cx,cy,x,y,w0,h0,_ = d
            tracks.append(Track(cx,cy,frame_idx,(x,y,x+w0,y+h0)))

        # 背景更新 & 前フレ更新
        bg = (1.0-ALPHA)*bg + ALPHA*gray                   # 背景 = 過去*係数 + 現在*係数
        gray_prev = gray                                   # 前フレ更新

        # ===== GOGOGO 判定 =====
        sum_area = sum(d[-1] for d in detections)          # このフレームの合計面積
        # トラック基準: 妥当な継続と速度があるトラックがいればGOに寄与
        has_moving_track = any(
            (not tr.done) and tr.is_valid() and (tr.velocity_px_per_s(fps) >= V_MIN)
            for tr in tracks
        )
        strong = (sum_area >= MIN_GO_AREA_STRONG) or has_moving_track  # 強い検出の定義
        win.append(1 if strong else 0)                       # 多数決ウィンドウ更新
        new_go_flag = (sum(win) >= WIN_M)                    # 5中2以上でGO

        # ===== 可視化(描画はI/Oの一部としてOpenCV使用) =====
        vis = frame.copy()
        if USE_CV_DRAW:
            guide_color = (255,255,0)                        # サイドバー(飾り)
            for i in range(3):
                y0 = GUIDE_MARGIN + i*(GUIDE_LEN + GUIDE_GAP)
                y1 = min(y0 + GUIDE_LEN, H-10)
                cv2.rectangle(vis, (8, y0), (8+GUIDE_THICK, y1), guide_color, -1)
                cv2.rectangle(vis, (W-8-GUIDE_THICK, y0), (W-8, y1), guide_color, -1)
            if DRAW_BLUE_ROI:
                cv2.rectangle(vis,(bx0,by0),(bx1,by1),(255,128,0),2)   # ROI可視化
            for d in detections:
                cx,cy,x,y,w0,h0,area = d
                cv2.rectangle(vis,(x-2,y-2),(x+w0+2,y+h0+2),(0,128,255),OBJ_RECT_THICK_OUT)  # 外枠
                cv2.rectangle(vis,(x,y),(x+w0,y+h0),(0,128,255),OBJ_RECT_THICK_IN)           # 内枠
            text = "GOGOGO" if display_go_flag else "STOP"           # 1フレ遅延表示
            banner_color = (0,0,255) if display_go_flag else (0,255,0)
            (tw, th), _ = cv2.getTextSize(text, BANNER_FONT, BANNER_SCALE, BANNER_THICK)
            cv2.rectangle(vis, (10,10), (10+tw+BANNER_PAD*2, 10+th+BANNER_PAD*2), banner_color, -1)
            cv2.putText(vis, text, (10+BANNER_PAD, 10+BANNER_PAD+th),
                        BANNER_FONT, BANNER_SCALE, (255,255,255), BANNER_THICK, cv2.LINE_AA)
        else:
            # NumPyだけで簡易矩形を描く(テキストなし)
            for d in detections:
                _,_,x,y,w0,h0,_ = d
                vis[y:y+2, x:x+w0] = (0,128,255)
                vis[y+h0-2:y+h0, x:x+w0] = (0,128,255)
                vis[y:y+h0, x:x+2] = (0,128,255)
                vis[y:y+h0, x+w0-2:x+w0] = (0,128,255)

        vw.write(vis)                                       # 書き出し
        if show and (frame_idx % disp_every == 0):          # 間引きプレビュー
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):           # qで中断
                break

        display_go_flag = new_go_flag                       # 1フレ遅延更新

    # 後処理: リソース解放
    cap.release(); vw.release(); cv2.destroyAllWindows()

    # 未終了トラックも暫定記録
    for tr in tracks:
        if not tr.done and tr.is_valid():
            finished_rows.append([tr.id, tr.start, tr.last, tr.dx_total(), 0.0])

    # CSV出力
    with open("tracks.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["id","start_frame","end_frame","dx_total_px","avg_speed_px_per_s"])
        w.writerows(finished_rows)
    with open("detections.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["frame","cx","cy","x","y","w","h","area","detected"])
        w.writerows(detection_rows)

    # 保存先を表示
    print("Saved video:", os.path.abspath(args.output))
    print("Saved tracks:", os.path.abspath("tracks.csv"))
    print("Saved detections:", os.path.abspath("detections.csv"))

# エントリポイント
if __name__ == "__main__":
    main()
