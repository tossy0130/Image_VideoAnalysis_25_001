import cv2
import numpy as np
import argparse
from collections import deque

# ============= パラメータ（必要に応じて調整） =============
W, H = 960, 540

# 肌色抽出（YCrCb）：照明変化に比較的強い
YCRCB_MIN = (0, 135, 85)     # (Y,Cr,Cb) 下限
YCRCB_MAX = (255, 180, 135)  # 上限

# 指先判定（凸性欠陥）
DEFECT_DEPTH_MIN = 12        # 欠陥の深さ（px）下限
FINGERTIP_ANGLE_MAX = 70.0   # 指先の鋭さ（角度上限, 度）
MIN_CONTOUR_AREA = 2500      # 小さすぎる輪郭を無視

# 関節の近似位置（指先→掌中心の線上を分割）
JOINT_RATIOS = [0.33, 0.66]  # MCP/PIP/DIP“っぽい”位置（2点）

# 平滑化
SMOOTH_ALPHA = 0.35          # 0に近いほどヌルヌル、1に近いほど生

KERNEL = np.ones((3,3), np.uint8)

# ============= ユーティリティ =============
def expo_smooth(prev, cur, alpha=SMOOTH_ALPHA):
    """指数移動平均の平滑化"""
    if prev is None: return cur
    return prev*(1-alpha) + cur*alpha

def angle_deg(a, b, c):
    """3点 a-b-c の∠b（度）"""
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    cosv = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def find_hand_keypoints(mask):
    """
    肌色マスクから最大輪郭を手として採用し、
    凸包＋凸性欠陥で指先候補、重心で掌中心、下端で手首近傍を推定
    """
    # ノイズ整形
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA:
        return None

    # 掌中心（重心）
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6: 
        return None
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    palm = np.array([cx, cy], dtype=np.float32)

    # 凸包 & 凸性欠陥
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None
    defects = cv2.convexityDefects(cnt, hull)

    fingertips = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            ps = cnt[s,0]; pe = cnt[e,0]; pf = cnt[f,0]
            depth = d/256.0  # OpenCVは×256スケール

            # 指間の谷が深く、形が鋭い（角度が小さい）とき、端点のどちらかを指先候補に
            if depth > DEFECT_DEPTH_MIN:
                ang = angle_deg(ps.astype(np.float32), pf.astype(np.float32), pe.astype(np.float32))
                if ang < FINGERTIP_ANGLE_MAX:
                    ds = np.linalg.norm(ps - palm)
                    de = np.linalg.norm(pe - palm)
                    tip = ps if ds > de else pe
                    fingertips.append(tip)

    # 近傍の重複統合（近い指先候補をマージ）
    merged = []
    used = [False]*len(fingertips)
    MERGE_DIST = 25
    for i, p in enumerate(fingertips):
        if used[i]: continue
        group = [p]
        used[i] = True
        for j, q in enumerate(fingertips):
            if used[j] or i==j: continue
            if np.linalg.norm(p - q) < MERGE_DIST:
                group.append(q); used[j] = True
        g = np.mean(np.array(group), axis=0).astype(np.int32)
        merged.append(g)

    # 手首（輪郭の最下点近傍を採用：簡易）
    wrist = cnt[cnt[:,:,1].argmax()][0].astype(np.int32)

    return {"contour": cnt, "palm": palm.astype(np.int32), "wrist": wrist, "tips": merged}

def joints_along_ray(tip, palm, ratios=JOINT_RATIOS):
    """指先→掌中心の線上に、比率で関節“っぽい”点を置く"""
    tip = tip.astype(np.float32); palm = palm.astype(np.float32)
    vec = palm - tip
    return [ (tip + r*vec).astype(np.int32) for r in ratios ]

def draw_hand_skeleton(vis, kp, prev_state):
    """推定キーポイントを平滑化しつつ描画"""
    palm = kp["palm"].astype(np.float32)
    wrist = kp["wrist"].astype(np.float32)
    tips  = [t.astype(np.float32) for t in kp["tips"]]

    # 平滑
    prev_state["palm"]  = expo_smooth(prev_state.get("palm"),  palm)
    prev_state["wrist"] = expo_smooth(prev_state.get("wrist"), wrist)

    tips_s = []
    prev_tips = prev_state.get("tips", [])
    for i, t in enumerate(tips):
        t_prev = prev_tips[i] if i < len(prev_tips) else None
        tips_s.append(expo_smooth(t_prev, t))
    prev_state["tips"] = tips_s

    # 描画
    p = prev_state["palm"].astype(np.int32)
    w = prev_state["wrist"].astype(np.int32)
    cv2.circle(vis, p, 6, (0,255,255), -1)     # 掌中心
    cv2.circle(vis, w, 6, (255,0,255), -1)     # 手首
    cv2.line(vis, tuple(p), tuple(w), (200,200,200), 3)

    for t in tips_s:
        if t is None: continue
        ti = t.astype(np.int32)
        cv2.circle(vis, ti, 5, (0,128,255), -1)   # 指先
        joints = joints_along_ray(t, prev_state["palm"])  # 擬似関節
        for j in joints:
            cv2.circle(vis, tuple(j), 4, (0,255,0), -1)
        pts = [ti] + joints + [p]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(vis, tuple(a), tuple(b), (0,200,0), 2)

# ============= メイン =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="入力動画パス（未指定ならWebカメラ）")
    ap.add_argument("--codec", default="avc1", help="出力コーデック（avc1/mp4v/XVIDなど）")
    ap.add_argument("--output", help="注釈付き出力動画パス（任意）")
    ap.add_argument("--no-preview", action="store_true", help="プレビューなしで実行")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if not args.input else args.input)
    if not cap.isOpened():
        print("動画/カメラを開けません"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = None
    if args.output:
        vw = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    prev_state = {}
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (W, H))

        # 肌色抽出（YCrCb）
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, YCRCB_MIN, YCRCB_MAX)

        # 推定
        kp = find_hand_keypoints(skin)

        # 可視化
        vis = frame.copy()

        # マスクの小窓（左上）
        mask_small = cv2.resize(skin, (W//4, H//4))
        vis[10:10+mask_small.shape[0], 10:10+mask_small.shape[1]] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        if kp is not None:
            cv2.drawContours(vis, [kp["contour"]], -1, (200,200,200), 1)
            draw_hand_skeleton(vis, kp, prev_state)
        else:
            prev_state.clear()

        if vw is not None:
            vw.write(vis)

        if not args.no_preview:
            cv2.imshow("hand joints (classic, no-ML)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if vw is not None: vw.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
