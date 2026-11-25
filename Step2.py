# -*- coding: utf-8 -*-
"""
Step2.py —— 基于 Step1 的输出，计算线虫的：轨迹、速度、摆动频率。

输入：
    step1_output/<视频名>/
        worm_trajectory.csv    (frame_index, seq_index, center_x, center_y, area, length, crop_file)
        worm_crops/worm_*.png  (逐帧裁剪图)

输出：
    每个视频文件夹内：
        worm_features.csv            逐帧：质心、头尾点、中线长度、宽度、弯曲角、曲率等
        worm_analysis_summary.csv    该视频汇总：平均/最大速度、体长、摆动频率、覆盖率、跳帧原因
    批量时（输入为 step1_output 整目录）：
        step2_output/batch_summary.csv  所有视频汇总合并成一张表

用法（窗口模式，直接运行即可）：
    python Step2.py

用法（命令行 / 无窗口）：
    python Step2.py --input step1_output/1-2
    python Step2.py --input step1_output/1-2 --um-per-px 2.0 --plot
    python Step2.py --input step1_output --um-per-px 2.0 --plot

参数：
    --input      可选；默认弹出窗口选择，未选则用脚本同目录下的 step1_output。
    --fps        可选；原始视频帧率。不填则自动读 worm_stats.csv 或按 frame_index 间隔检测。
    --um-per-px  像素到微米的标尺。不填则速度/长度只报像素单位。
    --plot       需要 matplotlib；没有则跳过画图。
"""

import argparse
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_FPS = 12.0          # Step1 的有效采样率
BODY_RESAMPLE = 41         # 中线重采样点数
FREQ_RANGE = (0.3, 4.0)    # 摆动频率搜索范围 (Hz)，线虫游动一般在 0.5~2 Hz
SPECTRUM_ALPHA = 0.05      # 频谱显著性阈值（整体误报率），越小越严格


# ============================================================ 中线/骨架 ============================================================
def zhang_suen_thinning(binary):
    """Zhang-Suen 骨架化，只依赖 numpy，不需要 opencv-contrib。"""
    img = (binary > 0).astype(np.uint8)
    while True:
        changed = False
        for phase in (1, 2):
            p = np.pad(img, 1, mode="constant")
            P2 = p[:-2, 1:-1]   # N
            P3 = p[:-2, 2:]     # NE
            P4 = p[1:-1, 2:]    # E
            P5 = p[2:, 2:]      # SE
            P6 = p[2:, 1:-1]    # S
            P7 = p[2:, :-2]     # SW
            P8 = p[1:-1, :-2]   # W
            P9 = p[:-2, :-2]    # NW
            neigh = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(neigh)

            # A: 顺时针 P2..P9 -> P2 中 0->1 的跳变次数
            A = np.zeros_like(P2, dtype=np.int32)
            seq = neigh + [P2]
            for a, b in zip(seq, seq[1:]):
                A += ((a == 0) & (b == 1)).astype(np.int32)

            cond_b = (B >= 2) & (B <= 6)
            cond_a = (A == 1)
            if phase == 1:
                cond_c = (P2 * P4 * P6 == 0)
                cond_d = (P4 * P6 * P8 == 0)
            else:
                cond_c = (P2 * P4 * P8 == 0)
                cond_d = (P2 * P6 * P8 == 0)

            remove = (img == 1) & cond_b & cond_a & cond_c & cond_d
            if remove.any():
                changed = True
                img = img & (~remove).astype(np.uint8)
        if not changed:
            break
    return img


def extract_centerline(skel):
    """从骨架提取中线：返回按顺序排列的点数组 (N,2)，坐标为 (x, y)。"""
    ys, xs = np.where(skel > 0)
    if len(xs) < 5:
        return None
    pts = set(zip(xs.tolist(), ys.tolist()))  # (x, y)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    adj = {p: [] for p in pts}
    for (x, y) in pts:
        for dx, dy in offs:
            q = (x + dx, y + dy)
            if q in pts:
                adj[(x, y)].append(q)

    endpoints = [p for p in pts if len(adj[p]) == 1]
    if not endpoints:
        endpoints = [next(iter(pts))]

    def bfs_dist(s):
        dist = {s: 0}
        prev = {s: None}
        dq = deque([s])
        while dq:
            cur = dq.popleft()
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    prev[nb] = cur
                    dq.append(nb)
        far = max(dist, key=lambda k: dist[k])
        return prev, far

    _, e1 = bfs_dist(endpoints[0])
    prev, e2 = bfs_dist(e1)
    path = []
    cur = e2
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path = path[::-1]  # e1 -> e2
    return np.asarray(path, dtype=np.float32)


def resample_polyline(pts, n):
    """沿弧长等距重采样为 n 个点。"""
    pts = np.asarray(pts, dtype=np.float32)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total <= 1e-6:
        return None
    s = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    for k in range(2):
        out[:, k] = np.interp(s, cum, pts[:, k])
    return out


def centerline_length(pts):
    pts = np.asarray(pts, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ============================================================ 分割 ============================================================
def _best_worm_contour(binary, h, w, min_area=40, max_area_frac=0.7, min_aspect=1.4):
    """从二值图中挑出最像线虫的连通域：按细长程度打分，返回 (score, cnt, area) 或 None。"""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area_frac * h * w:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect < min_aspect:
            continue
        per = cv2.arcLength(cnt, True)
        elong = per * per / (4 * np.pi * area) if area > 0 else 0.0  # 细长度，越大越细长
        score = aspect + 0.5 * min(elong, 20.0) + 0.5 * np.log10(area + 1.0)
        if best is None or score > best[0]:
            best = (score, cnt, area)
    return best


def segment_worm(gray, bg_gray=None):
    """分割线虫：优先普通 Otsu（可靠基线），Otsu 找不到时才用背景差兜底。
    返回 (mask, cnt, polarity, area)，失败返回 None。"""
    h, w = gray.shape
    if gray.std() < 1.5:
        return None

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def pick_from_binary(binary):
        return _best_worm_contour(binary, h, w)

    # 方法1：普通 Otsu（双极性）—— 可靠基线
    otsu = []
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for polarity in ("dark", "bright"):
        m = (gray <= thr).astype(np.uint8) if polarity == "dark" else (gray >= thr).astype(np.uint8)
        m = (m * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        c = pick_from_binary(m)
        if c is not None:
            score, cnt, area = c
            otsu.append((score, polarity, cnt, area))

    # 方法2：背景差（仅当 Otsu 完全找不到时使用）—— 去除舱壁/静态杂物
    bgdiff = []
    if bg_gray is not None:
        residual = cv2.absdiff(gray, bg_gray)
        residual = cv2.medianBlur(residual, 3)
        thr, _ = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if thr < 8:
            thr = 8
        _, binary = cv2.threshold(residual, thr, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        c = pick_from_binary(binary)
        if c is not None:
            score, cnt, area = c
            bgdiff.append((score, "bgdiff", cnt, area))

    cands = otsu if otsu else bgdiff
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    _, polarity, cnt, area = cands[0]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask, cnt, polarity, area


def identify_head_tail(centerline, dist_map):
    """线虫头部通常比尾部略宽。用端点附近的距离变换(半宽)判断头尾。"""
    n = len(centerline)
    frac = max(1, n // 6)

    def endpoint_width(idx):
        seg = centerline[:frac] if idx == 0 else centerline[n - frac:]
        xs = np.clip(seg[:, 0].astype(int), 0, dist_map.shape[1] - 1)
        ys = np.clip(seg[:, 1].astype(int), 0, dist_map.shape[0] - 1)
        return float(dist_map[ys, xs].mean() * 2.0)

    w0 = endpoint_width(0)
    w1 = endpoint_width(n - 1)
    if w0 >= w1:
        return 0, n - 1          # head_idx, tail_idx
    return n - 1, 0


def signed_mid_curvature(cl_ht):
    """身体中点的带符号曲率（度）。游动时它随摆动周期震荡，用于 FFT 求频率。"""
    n = len(cl_ht)
    t = np.arctan2(np.diff(cl_ht[:, 1]), np.diff(cl_ht[:, 0]))
    k = max(1, n // 8)
    d = t[n // 2 + k] - t[n // 2 - k]
    return float(np.degrees(wrap_pi(d)))


def head_tail_bend(cl_ht):
    """头-中点-尾 的带符号夹角（度），衡量整体弯曲（C 形明显）。"""
    n = len(cl_ht)
    head = cl_ht[0]
    tail = cl_ht[-1]
    mid = cl_ht[n // 2]
    v1 = head - mid
    v2 = tail - mid
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    return float(np.degrees(wrap_pi(a2 - a1)))


# ============================================================ 单帧分析 ============================================================
def analyze_crop(img, bg_crop=None):
    """返回 (结果dict, 跳过原因)。成功时 reason 为 None。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    seg = segment_worm(gray, bg_crop)
    if seg is None:
        return None, "无细长虫体(低对比/盘曲/被切断)"
    mask, cnt, polarity, area = seg

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, "轮廓矩异常"
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    skel = zhang_suen_thinning(mask)
    cl = extract_centerline(skel)
    if cl is None or len(cl) < 10:
        return None, "骨架/中线过短"

    length_px = centerline_length(cl)
    cl = resample_polyline(cl, BODY_RESAMPLE)
    if cl is None:
        return None, "中线退化"

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    head_idx, tail_idx = identify_head_tail(cl, dist)
    cl_ht = cl if head_idx == 0 else cl[::-1]

    head_local = cl_ht[0]
    tail_local = cl_ht[-1]
    width_px = float(dist[dist > 0].mean() * 2.0) if dist.max() > 0 else 0.0

    return {
        "local_cx": float(cx), "local_cy": float(cy),
        "head_local": head_local, "tail_local": tail_local,
        "length_px": length_px, "width_px": width_px,
        "bend_head_tail": head_tail_bend(cl_ht),
        "curv_mid": signed_mid_curvature(cl_ht),
        "polarity": polarity, "area": float(area),
    }, None


# ============================================================ 汇总分析 ============================================================
def smooth_ma(x, window):
    if window <= 1 or len(x) < window:
        return np.asarray(x, dtype=float)
    k = window // 2
    y = np.convolve(x, np.ones(window) / window, mode="same")
    y = np.asarray(y, dtype=float)
    # 边界用原值，避免卷积边界失真
    y[:k] = x[:k]
    y[-k:] = x[-k:]
    return y


def estimate_frequency(t, signal, fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1]):
    """对非均匀采样的弯曲信号做 FFT，返回主频(Hz)。"""
    if signal is None or len(signal) < 16:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    t_grid = np.arange(t[0], t[-1], dt)
    s = np.interp(t_grid, t, signal)
    # 去线性趋势
    coef = np.polyfit(t_grid, s, 1)
    s = s - np.polyval(coef, t_grid)
    s = s - s.mean()

    n = len(s)
    if n < 16:
        return None
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(s * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    sel = (freqs >= fmin) & (freqs <= fmax)
    if not sel.any():
        return None
    peak = freqs[sel][np.argmax(spec[sel])]
    return float(peak)


def process_folder(folder, fps, um_per_px, do_plot, use_bg=True):
    folder = Path(folder)
    traj_csv = folder / "worm_trajectory.csv"
    crop_dir = folder / "worm_crops"
    if not traj_csv.exists():
        return {"folder": str(folder), "error": "缺少 worm_trajectory.csv"}

    df = pd.read_csv(traj_csv)
    df = df.sort_values("frame_index").reset_index(drop=True)

    stats_csv = folder / "worm_stats.csv"

    # fps 优先级：显式参数 > worm_stats.csv 里的 fps > 按 frame_index 间隔自动检测
    if fps is None:
        if stats_csv.exists():
            try:
                st = pd.read_csv(stats_csv)
                if "fps" in st.columns and len(st) > 0 and not pd.isna(st["fps"].iloc[0]):
                    fps = float(st["fps"].iloc[0])
            except Exception:
                pass
    if fps is None:
        steps = np.diff(df["frame_index"].values)
        steps = steps[steps > 0]
        if len(steps) > 0:
            fps = round(float(np.median(steps)) * TARGET_FPS)  # step = round(fps/12)
        else:
            fps = 60.0

    # 背景图（Step1 重跑后生成 background.png），用于背景差，去除舱壁/静态杂物
    bg_full = None
    if use_bg:
        bg_file = folder / "background.png"
        if bg_file.exists():
            bg_full = cv2.imread(str(bg_file), cv2.IMREAD_GRAYSCALE)

    # 观察仓圆参数（用于轨迹图叠加圆）
    circle = None
    if stats_csv.exists():
        try:
            st = pd.read_csv(stats_csv)
            cols = {"chamber_center_x", "chamber_center_y", "chamber_radius_px"}
            if cols.issubset(st.columns) and len(st) > 0:
                cx = st["chamber_center_x"].iloc[0]
                cy = st["chamber_center_y"].iloc[0]
                cr = st["chamber_radius_px"].iloc[0]
                if not (pd.isna(cx) or pd.isna(cy) or pd.isna(cr)):
                    circle = (float(cx), float(cy), float(cr))
        except Exception:
            pass

    rows = []
    n_ok = 0
    n_skip = 0
    skip_reasons = {}
    for _, r in df.iterrows():
        crop_file = crop_dir / r["crop_file"]
        # 轨迹点始终用 Step1 记录的质心，跳帧也不会丢轨迹
        row = {
            "frame_index": int(r["frame_index"]),
            "seq_index": int(r["seq_index"]),
            "centroid_x": float(r["center_x"]),
            "centroid_y": float(r["center_y"]),
            "head_x": np.nan, "head_y": np.nan,
            "tail_x": np.nan, "tail_y": np.nan,
            "length_px": np.nan, "width_px": np.nan,
            "bend_head_tail_deg": np.nan, "curv_mid_deg": np.nan,
            "polarity": None,
        }

        if not crop_file.exists():
            n_skip += 1
            skip_reasons["文件缺失"] = skip_reasons.get("文件缺失", 0) + 1
            rows.append(row); continue
        img = cv2.imread(str(crop_file))
        if img is None:
            n_skip += 1
            skip_reasons["读图失败"] = skip_reasons.get("读图失败", 0) + 1
            rows.append(row); continue
        H, W = img.shape[:2]

        # 局部 -> 全局：优先用 Step1 写入的精确裁剪原点，否则用中心回推
        if "crop_x" in r.index and not pd.isna(r["crop_x"]):
            x1 = float(r["crop_x"])
            y1 = float(r["crop_y"])
        else:
            x1 = float(r["center_x"]) - W / 2.0
            y1 = float(r["center_y"]) - H / 2.0

        # 背景切片（有背景图则做背景差，去除舱壁/静态杂物）
        bg_crop = None
        if bg_full is not None:
            xi = max(0, min(int(round(x1)), bg_full.shape[1] - W))
            yi = max(0, min(int(round(y1)), bg_full.shape[0] - H))
            bg_crop = bg_full[yi:yi + H, xi:xi + W]

        res, reason = analyze_crop(img, bg_crop)
        if res is None:
            n_skip += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            rows.append(row); continue
        n_ok += 1

        row["head_x"] = res["head_local"][0] + x1
        row["head_y"] = res["head_local"][1] + y1
        row["tail_x"] = res["tail_local"][0] + x1
        row["tail_y"] = res["tail_local"][1] + y1
        row["length_px"] = res["length_px"]
        row["width_px"] = res["width_px"]
        row["bend_head_tail_deg"] = res["bend_head_tail"]
        row["curv_mid_deg"] = res["curv_mid"]
        row["polarity"] = res["polarity"]
        rows.append(row)

    if not rows:
        return {"folder": str(folder), "error": "没有可分析的裁剪图"}

    f = pd.DataFrame(rows)
    f = f.sort_values("frame_index").reset_index(drop=True)

    # ---- 头尾时序一致性：让头尾标签沿时间连续，避免符号乱跳（跳过的帧 head 为 NaN，跳过） ----
    for i in range(1, len(f)):
        if pd.isna(f["head_x"].iloc[i]) or pd.isna(f["head_x"].iloc[i - 1]):
            continue
        dh_same = np.hypot(f["head_x"].iloc[i] - f["head_x"].iloc[i - 1],
                           f["head_y"].iloc[i] - f["head_y"].iloc[i - 1]) + \
                  np.hypot(f["tail_x"].iloc[i] - f["tail_x"].iloc[i - 1],
                           f["tail_y"].iloc[i] - f["tail_y"].iloc[i - 1])
        dh_swap = np.hypot(f["head_x"].iloc[i] - f["tail_x"].iloc[i - 1],
                           f["head_y"].iloc[i] - f["tail_y"].iloc[i - 1]) + \
                  np.hypot(f["tail_x"].iloc[i] - f["head_x"].iloc[i - 1],
                           f["tail_y"].iloc[i] - f["head_y"].iloc[i - 1])
        if dh_swap < dh_same:
            hx, tx = f.loc[i, "head_x"], f.loc[i, "tail_x"]
            hy, ty = f.loc[i, "head_y"], f.loc[i, "tail_y"]
            f.loc[i, "head_x"], f.loc[i, "tail_x"] = tx, hx
            f.loc[i, "head_y"], f.loc[i, "tail_y"] = ty, hy
            f.loc[i, "bend_head_tail_deg"] = -f.loc[i, "bend_head_tail_deg"]
            f.loc[i, "curv_mid_deg"] = -f.loc[i, "curv_mid_deg"]

    # ---- 时间轴 ----
    t = f["frame_index"].values / fps

    # ---- 轨迹（质心，做轻度平滑；保持原样） ----
    sm_x = smooth_ma(f["centroid_x"].values, 5)
    sm_y = smooth_ma(f["centroid_y"].values, 5)

    # ---- 速度：先对质心中值滤波（去孤立跳变），再按真实 dt 求速度 ----
    vx = pd.Series(f["centroid_x"].values).rolling(5, center=True, min_periods=1).median().values
    vy = pd.Series(f["centroid_y"].values).rolling(5, center=True, min_periods=1).median().values
    dx = np.diff(vx)
    dy = np.diff(vy)
    d_frame = np.diff(f["frame_index"].values)
    dt_sec = np.maximum(d_frame / fps, 1e-6)
    speed_px = np.concatenate([[np.nan], np.sqrt(dx ** 2 + dy ** 2) / dt_sec])
    # 平滑速度曲线（约 1 秒 = 12 个采样点）
    speed_smooth = pd.Series(speed_px).rolling(12, center=True, min_periods=1).mean().values

    f["time_s"] = t
    f["traj_x"] = sm_x
    f["traj_y"] = sm_y
    f["speed_px_s"] = speed_px
    f["speed_smooth_px_s"] = speed_smooth
    if um_per_px is not None:
        f["speed_um_s"] = speed_px * um_per_px
        f["speed_smooth_um_s"] = speed_smooth * um_per_px
        f["length_um"] = f["length_px"] * um_per_px

    f.to_csv(folder / "worm_features.csv", index=False)

    # ---- 摆动频率：仅用成功提取到中线曲率的帧 ----
    curv_all = f["curv_mid_deg"].values
    bend_all = f["bend_head_tail_deg"].values
    valid = ~np.isnan(curv_all)
    curv = curv_all[valid]
    bend = bend_all[valid]
    t_valid = t[valid]
    f_curv = estimate_frequency(t_valid, curv)
    f_bend = estimate_frequency(t_valid, bend)

    # 幅度用去趋势后的标准差估计
    def amplitude(tv, sig):
        tv = np.asarray(tv); sig = np.asarray(sig)
        if len(sig) < 8:
            return float("nan")
        dt = np.median(np.diff(tv))
        tg = np.arange(tv[0], tv[-1], dt)
        s = np.interp(tg, tv, sig)
        s = s - np.polyval(np.polyfit(tg, s, 1), tg)
        return float(np.std(s))

    freq_hz = f_curv if f_curv is not None else f_bend
    ampl = amplitude(t_valid, curv)

    duration_s = float((f["frame_index"].iloc[-1] - f["frame_index"].iloc[0]) / fps)
    path_len_px = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    skip_reasons_str = "; ".join(f"{k}:{v}" for k, v in sorted(skip_reasons.items(), key=lambda x: -x[1]))

    summary = {
        "folder": folder.name,
        "n_frames": int(len(f)),
        "n_analyzed": n_ok,
        "n_skipped": n_skip,
        "coverage": round(n_ok / max(1, n_ok + n_skip), 4),
        "duration_s": duration_s,
        "fps": fps,
        "mean_speed_px_s": float(np.nanmean(speed_px)),
        "median_speed_px_s": float(np.nanmedian(speed_px)),
        "max_speed_px_s": float(np.nanmax(speed_px)),
        "mean_length_px": float(f["length_px"].mean()),
        "mean_width_px": float(f["width_px"].mean()),
        "path_length_px": path_len_px,
        "undulation_freq_hz": freq_hz,
        "undulation_amplitude_deg": ampl,
        "skip_reasons": skip_reasons_str,
    }
    if um_per_px is not None:
        summary["mean_speed_um_s"] = float(np.nanmean(speed_px) * um_per_px)
        summary["max_speed_um_s"] = float(np.nanmax(speed_px) * um_per_px)
        summary["mean_length_um"] = float(f["length_px"].mean() * um_per_px)

    if circle is not None:
        summary["chamber_center_x"] = circle[0]
        summary["chamber_center_y"] = circle[1]
        summary["chamber_radius_px"] = circle[2]

    pd.DataFrame([summary]).to_csv(folder / "worm_analysis_summary.csv", index=False)

    if do_plot:
        _plot(folder, f, t, freq_hz, um_per_px, circle)

    return summary


# ============================================================ 画图（可选） ============================================================
def cycle_frequencies(t, signal, fmin=FREQ_RANGE[0], fmax=5.0, smooth_win=5, hyst_frac=0.1):
    """逐周期估计瞬时频率：在平滑信号上检测向上过零点（带迟滞），相邻两次间隔=一个周期。
    返回 (tc, fc, durations)：周期中点时间、周期对应频率、该周期时长；数据不足返回 None。"""
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    valid = ~np.isnan(s)
    if valid.sum() < 8:
        return None
    # 去线性趋势 + 去均值（只用有效点拟合）
    coef = np.polyfit(t[valid], s[valid], 1)
    s = s - np.polyval(coef, t)
    s = s - np.nanmean(s)
    # 轻度平滑降噪
    s = pd.Series(s).rolling(smooth_win, center=True, min_periods=1).mean().values
    h = hyst_frac * (np.nanstd(s) + 1e-9)

    crossings = []
    armed = False
    for i in range(1, len(s)):
        a, b = s[i - 1], s[i]
        if np.isnan(a) or np.isnan(b):
            continue
        if b <= -h:
            armed = True
        elif armed and b >= h:
            armed = False
            denom = b - a
            frac = (-a / denom) if denom != 0 else 0.0
            frac = min(max(frac, 0.0), 1.0)
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))

    if len(crossings) < 2:
        return None
    crossings = np.asarray(crossings)
    periods = np.diff(crossings)
    with np.errstate(divide="ignore", invalid="ignore"):
        freqs = 1.0 / periods
    ok = (freqs >= fmin) & (freqs <= fmax)
    if ok.sum() < 1:
        return None
    tc = (crossings[:-1] + crossings[1:]) / 2.0
    return tc[ok], freqs[ok], periods[ok]


def _plot(folder, f, t, freq_hz, um_per_px, circle=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (未安装 matplotlib，跳过绘图)")
        return

    unit = "um/s" if um_per_px else "px/s"
    speed = f["speed_um_s"].values if um_per_px is not None else f["speed_px_s"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(f["traj_x"], f["traj_y"], lw=0.8)
    axes[0, 0].set_title("trajectory (centroid, frame coords)"); axes[0, 0].set_aspect("equal")
    if circle is not None:
        cx, cy, r = circle
        axes[0, 0].add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor="red", linestyle="--", linewidth=1.2))
    sp_sm = f["speed_smooth_um_s"].values if um_per_px is not None else f["speed_smooth_px_s"].values
    axes[0, 1].plot(t, speed, lw=0.5, alpha=0.35, label="instantaneous")
    axes[0, 1].plot(t, sp_sm, lw=1.4, color="tab:orange", label="smoothed (~1s)")
    axes[0, 1].set_title(f"speed ({unit})")
    axes[0, 1].legend(fontsize=7)

    # 身体弯曲：原始带符号（细、淡）+ 平滑趋势（粗）+ |中点曲率| 幅度包络
    bend_raw = f["bend_head_tail_deg"].values
    bend_smooth = pd.Series(bend_raw).rolling(9, center=True, min_periods=3).mean().values
    axes[1, 0].plot(t, bend_raw, lw=0.6, alpha=0.35, color="tab:orange", label="head-tail bend (raw)")
    axes[1, 0].plot(t, bend_smooth, lw=1.4, color="tab:orange", label="bend (smoothed)")
    axes[1, 0].plot(t, np.abs(f["curv_mid_deg"].values), lw=1.0, color="tab:blue", alpha=0.8, label="|mid curvature|")
    axes[1, 0].axhline(0, color="gray", lw=0.6, ls="--")
    axes[1, 0].set_title("body bend (deg)"); axes[1, 0].legend(fontsize=7)

    # 频谱：只用有曲率数据的帧（跳过的帧为 NaN，不能直接插值）
    cv_all = f["curv_mid_deg"].values
    mask = ~np.isnan(cv_all)
    tv, cv = t[mask], cv_all[mask]
    if len(cv) >= 16:
        dt = np.median(np.diff(tv))
        tg = np.arange(tv[0], tv[-1], dt)
        s = np.interp(tg, tv, cv)
        s = s - np.polyval(np.polyfit(tg, s, 1), tg)
        s = s - s.mean()
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        spec_sm = pd.Series(spec).rolling(5, center=True, min_periods=1).mean().values
        fr = np.fft.rfftfreq(len(s), d=dt)

        # 线性纵轴 + 白噪声显著性阈值（Bonferroni，整体误报率 SPECTRUM_ALPHA）
        noise_mean = float(np.median(spec) / np.log(2.0))   # 指数分布：median = ln2·mean
        m_eff = max(2, len(spec) // 5)                      # 平滑后有效独立频点数
        thr = noise_mean * np.log(m_eff / SPECTRUM_ALPHA)

        axes[1, 1].plot(fr, spec, lw=0.6, alpha=0.3, label="periodogram")
        axes[1, 1].plot(fr, spec_sm, lw=1.2, label="smoothed (5-bin)")
        axes[1, 1].axhline(thr, color="red", ls="--", lw=1.0,
                           label=f"significance α={SPECTRUM_ALPHA}")
        above = spec_sm > thr
        if np.any(above):
            axes[1, 1].plot(fr[above], spec_sm[above], "o", ms=3, color="red", label="peaks")
        # 峰突出度（峰/噪声中位数），超过阈值与否都给出数字，避免“没超阈值=没特征”的误判
        if np.any(spec > 0):
            peak_ratio = float(np.max(spec_sm) / (noise_mean + 1e-12))
            axes[1, 1].text(0.98, 0.95, f"peak/noise≈{peak_ratio:.1f}×",
                            transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=8)
        axes[1, 1].set_xlim(0, 5)
        axes[1, 1].set_xlabel("frequency (Hz)")
        axes[1, 1].set_ylabel("relative power (linear)")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].set_title(f"power spectrum of mid-curvature (peak={freq_hz:.2f} Hz)" if freq_hz else "power spectrum of mid-curvature")
    else:
        axes[1, 1].text(0.5, 0.5, "insufficient valid frames", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("power spectrum of mid-curvature")

    fig.tight_layout()
    out = folder / "worm_analysis.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  图已保存: {out}")

    # ---- 逐周期频率：随时间变化 + 各频率时间占比（单独一张图） ----
    cyc = cycle_frequencies(t, f["curv_mid_deg"].values)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
    if cyc is not None:
        tc, fc, dur = cyc
        ax2[0].step(tc, fc, where="post", lw=1.0, color="tab:green")
        ax2[0].set_xlabel("time (s)")
        ax2[0].set_ylabel("frequency (Hz)")
        ax2[0].set_title(f"per-cycle frequency over time (mean={float(np.mean(fc)):.2f} Hz)")

        bins = np.arange(0.3, 5.0 + 1e-9, 0.1)
        hist, edges = np.histogram(fc, bins=bins, weights=dur)
        total = hist.sum()
        if total > 0:
            ax2[1].bar(edges[:-1], hist / total * 100.0, width=0.1, align="edge", color="tab:green")
        ax2[1].set_xlabel("frequency (Hz)")
        ax2[1].set_ylabel("time share (%)")
        ax2[1].set_title("time share per frequency (per-cycle)")
    else:
        ax2[0].text(0.5, 0.5, "not enough cycles", ha="center", va="center", transform=ax2[0].transAxes)
        ax2[1].text(0.5, 0.5, "not enough cycles", ha="center", va="center", transform=ax2[1].transAxes)
        ax2[0].set_title("per-cycle frequency over time")
        ax2[1].set_title("time share per frequency (per-cycle)")
    fig2.tight_layout()
    out2 = folder / "worm_frequency.png"
    fig2.savefig(out2, dpi=110)
    plt.close(fig2)
    print(f"  图已保存: {out2}")


# ============================================================ 目录与批量 ============================================================
def _resolve_folders(root):
    if (root / "worm_trajectory.csv").exists():
        return [root]
    return sorted([d for d in root.iterdir() if d.is_dir() and (d / "worm_trajectory.csv").exists()])


def _step2_dir(root):
    """批量汇总 CSV 放哪：与 step1_output 同级、命名 step2_output。"""
    if (root / "worm_trajectory.csv").exists():
        return root.parent.parent / "step2_output"   # step1_output/<video> -> 上两级
    return root.parent / "step2_output"              # step1_output -> 上一级


def batch_filename(name=None):
    """批次汇总 CSV 文件名：给了名字就清洗后用，否则按时间（精确到分钟）。"""
    import re
    from datetime import datetime
    if name and str(name).strip():
        base = re.sub(r'[\\/:*?"<>|]', "_", str(name).strip())
        if base:
            return base if base.lower().endswith(".csv") else base + ".csv"
    return "batch_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv"


def _run(root_path, fps, um_per_px, do_plot, use_bg=True, batch_name=None):
    root = Path(root_path)
    if not root.exists():
        return None, f"路径不存在: {root}"

    folders = _resolve_folders(root)
    if not folders:
        return None, "未找到 worm_trajectory.csv，请检查所选目录。"

    step2_dir = _step2_dir(root)
    step2_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fd in folders:
        print(f"\n处理: {fd}")
        res = process_folder(fd, fps, um_per_px, do_plot, use_bg)
        if "error" in res:
            print(f"  ❌ {res['error']}")
            continue
        summaries.append(res)

    batch_csv = None
    if summaries:
        batch_df = pd.DataFrame(summaries)
        batch_csv = step2_dir / batch_filename(batch_name)
        batch_df.to_csv(batch_csv, index=False)

    return summaries, (str(batch_csv) if batch_csv else None)


# ============================================================ 命令行 ============================================================
def _print_summary(summaries, um_per_px, batch_csv):
    unit = "um/s" if um_per_px else "px/s"
    for s in summaries:
        print(f"\n—— {s['folder']} ——")
        print(f"  分析 {s['n_analyzed']} 帧，跳过 {s['n_skipped']} 帧，覆盖率 {s['coverage']*100:.1f}%")
        print(f"  时长 {s['duration_s']:.1f} s")
        print(f"  速度: 平均 {s['mean_speed_px_s']:.1f} px/s，最大 {s['max_speed_px_s']:.1f} px/s"
              + (f" (平均 {s['mean_speed_um_s']:.1f} {unit})" if um_per_px else ""))
        print(f"  体长: 平均 {s['mean_length_px']:.1f} px"
              + (f" = {s['mean_length_um']:.1f} um" if um_per_px else ""))
        if s["undulation_freq_hz"]:
            print(f"  摆动频率: {s['undulation_freq_hz']:.2f} Hz，幅度 {s['undulation_amplitude_deg']:.1f} deg")
        else:
            print("  摆动频率: 未能估计（有效帧太少）")
        if s["skip_reasons"]:
            print(f"  跳帧原因: {s['skip_reasons']}")
    if batch_csv:
        print(f"\n批量汇总已写入: {batch_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="step1_output 或其中的单个视频文件夹")
    ap.add_argument("--fps", type=float, default=None, help="原始视频帧率；不填自动检测")
    ap.add_argument("--um-per-px", type=float, default=None, help="像素->微米标尺")
    ap.add_argument("--plot", action="store_true", help="输出图表")
    ap.add_argument("--no-bg", action="store_true", help="禁用背景差（用于对比诊断）")
    ap.add_argument("--batch-name", default=None, help="批次汇总 CSV 名称；留空按时间命名")
    args = ap.parse_args()

    if args.input or args.fps or args.um_per_px or args.plot or args.no_bg or args.batch_name:
        # 有命令行参数 -> 无窗口模式
        root_path = args.input or (Path(__file__).resolve().parent / "step1_output")
        summaries, batch_csv = _run(root_path, args.fps, args.um_per_px, args.plot,
                                    use_bg=not args.no_bg, batch_name=args.batch_name)
        if summaries is None:
            print(batch_csv or "出错")
            return
        _print_summary(summaries, args.um_per_px, batch_csv)
    else:
        run_gui()


# ============================================================ 窗口模式 ============================================================
def run_gui():
    from tkinter import messagebox
    from ConfigWindow import ask_config

    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "step1_output"
    cfg = ask_config(default_input_dir=default_dir, default_output_dir=None,
                     title="Step2 - 参数设置",
                     show_video_list=False, show_scale=True, show_bg=True,
                     show_plot=True, show_dump=False, show_fps=False, show_batch_name=True)
    if cfg is None:
        return

    um_per_px = cfg["um_per_px"]
    summaries, batch_csv = _run(cfg["input"], cfg["fps"], um_per_px, cfg["plot"],
                                cfg["use_bg"], batch_name=cfg.get("batch_name"))
    if summaries is None:
        messagebox.showerror("出错", batch_csv or "处理失败")
        return

    _print_summary(summaries, um_per_px, batch_csv)

    unit = "µm/s" if um_per_px else "px/s"
    lines = []
    for s in summaries:
        freq = f"{s['undulation_freq_hz']:.2f} Hz" if s["undulation_freq_hz"] else "未估计"
        lines.append(
            f"{s['folder']}:\n"
            f"  分析 {s['n_analyzed']} 帧 / 跳过 {s['n_skipped']} 帧 (覆盖率 {s['coverage']*100:.1f}%)\n"
            f"  平均速度 {s['mean_speed_px_s']:.1f} px/s"
            + (f" = {s['mean_speed_um_s']:.1f} {unit}\n" if um_per_px else "\n")
            + f"  摆动频率 {freq}"
        )
    tail = f"\n\n批量汇总已写入:\n{batch_csv}" if batch_csv else ""
    messagebox.showinfo("完成", "\n\n".join(lines) + tail)


if __name__ == "__main__":
    main()
