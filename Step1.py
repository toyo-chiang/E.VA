import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ================= 可调参数 =================
TARGET_FPS = 12
CROP_SCALE = 1.6
TRIM_SECONDS = 5

# 裁剪图保存格式：PNG 无损但体积大（每帧存完整背景+噪声）；
# JPEG 体积约为 PNG 的 1/5~1/10，适合上千帧的批量输出。
CROP_SAVE_AS_JPEG = True      # 想回退无损 PNG 就改成 False
CROP_JPEG_QUALITY = 92        # 1~100，越大越清晰、体积越大

# ================= 鼠标交互 =================
drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0)
g_allow_draw = False

def put_chinese_text(img, text_lines, position, text_color=(0, 255, 0), text_size=28):
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    font = None
    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try:
            font = ImageFont.truetype(font_path, text_size); break
        except IOError:
            continue
    if font is None: font = ImageFont.load_default()
    x, y = position
    for line in text_lines:
        draw.text((x, y), line, font=font, fill=text_color)
        y += text_size + 8
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def mouse_draw_rect(event, x, y, flags, param):
    global drawing, ix, iy, rect, g_allow_draw
    if not g_allow_draw: return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; ix, iy = x, y; rect = (x, y, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing: rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False; rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))

def is_window_closed(win_name):
    try: return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
    except: return True

def auto_detect_chamber(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
                               param1=50, param2=30,
                               minRadius=int(min(h, w)*0.25), maxRadius=int(min(h, w)*0.95))
    if circles is not None:
        c = np.uint16(np.around(circles))
        return c[0, 0][0], c[0, 0][1], c[0, 0][2]
    return None

# ================= 单行进度条 =================
class SingleLineProgress:
    """
    单行刷新的终端进度条。
    - 用 \r 覆盖同行，不换行
    - 各数字段固定宽度，覆盖后无残留
    - 节流刷新，避免高频 flush 拖慢处理
    """
    def __init__(self, total, prefix="", bar_width=28, min_interval=0.05):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.bar_width = bar_width
        self.min_interval = min_interval
        self.start = time.time()
        self.last_render = 0.0
        self.last_len = 0

    def _render_line(self, n):
        frac = min(1.0, n / self.total)
        filled = int(self.bar_width * frac)
        if filled >= self.bar_width:
            bar = "=" * self.bar_width
        elif filled <= 0:
            bar = " " * self.bar_width
        else:
            bar = "=" * (filled - 1) + ">" + " " * (self.bar_width - filled)
        elapsed = time.time() - self.start
        fps = n / elapsed if elapsed > 1e-6 else 0.0
        if fps > 0 and n > 0:
            eta = int((self.total - n) / fps)
            eta_str = f"{eta:>6d}s"
        else:
            eta_str = "   --:--"
        return (f"\r{self.prefix}[{bar}] {frac*100:5.1f}% | "
                f"{n:>6}/{self.total:<6} | {fps:6.1f} fps | ETA {eta_str}")

    def update(self, n, force=False):
        now = time.time()
        if not force and (now - self.last_render) < self.min_interval:
            return
        self.last_render = now
        line = self._render_line(n)
        pad = max(0, self.last_len - len(line))
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
        self.last_len = len(line)

    def done(self):
        self.update(self.total, force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

# ================= 脊线增强 =================
def compute_ridge_response(gray_f, dark_worm=True, sigmas=(1.5, 2.5)):
    response = np.zeros_like(gray_f, dtype=np.float32)
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(gray_f, (0, 0), sigma)
        Dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) / 8.0
        Dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) / 8.0
        Dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        tmp = np.sqrt(((Dxx - Dyy) * 0.5) ** 2 + Dxy ** 2)
        lam2 = (Dxx + Dyy) * 0.5 + tmp
        lam1 = (Dxx + Dyy) * 0.5 - tmp
        resp = (np.maximum(lam2, 0) if dark_worm else np.maximum(-lam2, 0)) * sigma ** 2
        is_line = np.abs(lam1) < np.abs(lam2) * 0.6
        response = np.maximum(response, resp * is_line.astype(np.float32))
    return response

# ================= 候选检测 =================
def detect_candidates(gray, bg, safe_mask, params, roi=None):
    if roi is not None:
        x1, y1, x2, y2 = roi
    else:
        x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]
    region = gray[y1:y2, x1:x2]
    mask_r = safe_mask[y1:y2, x1:x2]
    if region.size == 0: return []
    gray_f = region.astype(np.float32)
    if bg is not None:
        bg_r = bg[y1:y2, x1:x2].astype(np.float32)
        diff = np.maximum(bg_r - gray_f, 0) if params["dark_worm"] else np.maximum(gray_f - bg_r, 0)
    else:
        local_bg = cv2.GaussianBlur(gray_f, (0, 0), 10)
        diff = np.maximum(local_bg - gray_f, 0) if params["dark_worm"] else np.maximum(gray_f - local_bg, 0)
    ridge = compute_ridge_response(gray_f, dark_worm=params["dark_worm"])
    d_scale = np.percentile(diff[diff > 0], 99) if np.any(diff > 0) else 1.0
    r_scale = np.percentile(ridge[ridge > 0], 99) if np.any(ridge > 0) else 1.0
    diff_n = np.clip(diff / (d_scale + 1e-6), 0, 1)
    ridge_n = np.clip(ridge / (r_scale + 1e-6), 0, 1)
    fused = (np.sqrt(diff_n * ridge_n) * 255).astype(np.uint8)
    fused = cv2.bitwise_and(fused, mask_r)
    masked_vals = fused[mask_r > 0]
    if len(masked_vals) < 100: return []
    thr_val, _ = cv2.threshold(masked_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr_val < 25: thr_val = 30
    _, binary = cv2.threshold(fused, thr_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_area"] or area > params["max_area"]: continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        length = max(bw, bh)
        if length < params["min_length"] * 0.5: continue
        # 质心（轮廓矩）比外接框中心更稳，用于轨迹/速度
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        candidates.append({
            "bbox": (bx + x1, by + y1, bw, bh),
            "area": area,
            "length": length,
            "centroid": (ccx + x1, ccy + y1),
        })
    return candidates

def score_candidate(cand, pred_pos, params):
    if params["target_area"] is None: return 0.0
    ratio = cand["area"] / params["target_area"]
    if ratio < 0.3 or ratio > 3.0: return 0.0
    area_score = min(ratio, 1.0 / ratio)
    lratio = cand["length"] / params["target_length"]
    if lratio < 0.5 or lratio > 2.0: return 0.0
    len_score = min(lratio, 1.0 / lratio)
    dist_score = 1.0
    if pred_pos is not None:
        bx, by, bw, bh = cand["bbox"]
        d = np.hypot(bx + bw/2 - pred_pos[0], by + bh/2 - pred_pos[1])
        dist_score = np.exp(-d / max(params["target_length"] * 2, 20))
    return area_score * len_score * dist_score

# ================= 简化跟踪器 =================
class Tracker:
    def __init__(self):
        self.pos = None
        self.vel = np.zeros(2)
        self.miss = 0
    def predict(self):
        return None if self.pos is None else self.pos + self.vel
    def update(self, pos):
        pos = np.array(pos, dtype=float)
        if self.pos is None:
            self.pos = pos
        else:
            self.vel = 0.6 * self.vel + 0.4 * (pos - self.pos)
            self.pos = pos
        self.miss = 0
    def mark_miss(self):
        self.miss += 1
        if self.miss > 15:
            self.pos = None; self.vel = np.zeros(2); self.miss = 0

def clip_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))

# ================= 交互学习 =================
def interactive_learning(video_path):
    global rect, g_allow_draw
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total <= 0: cap.release(); return None
    if total < fps * TRIM_SECONDS * 2 + 10:
        start_f, end_f = 0, total
    else:
        start_f = int(fps * TRIM_SECONDS)
        end_f = int(total - fps * TRIM_SECONDS)
    print("缓存测试帧...")
    cached_frames = []
    for _ in range(30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(start_f, max(start_f+1, end_f)))
        ret, f = cap.read()
        if ret: cached_frames.append(f)
    if not cached_frames: cap.release(); return None
    scale = 1200 / w if w > 1200 else 1.0

    # ===== 步骤1：确认观察仓 =====
    frame_disp_1 = cv2.resize(cached_frames[0], (int(w*scale), int(h*scale))) if scale < 1.0 else cached_frames[0].copy()
    auto_circle = auto_detect_chamber(frame_disp_1)
    win_name_1 = "Step 1: 确认观察仓边界"
    cv2.namedWindow(win_name_1, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win_name_1, mouse_draw_rect)
    circle_params = None
    mode_1_manual = False if auto_circle else True
    rect = (0, 0, 0, 0)
    g_allow_draw = True
    while True:
        if is_window_closed(win_name_1): cap.release(); return None
        temp_img = frame_disp_1.copy()
        if not mode_1_manual and auto_circle is not None:
            ccx, ccy, cr = auto_circle
            cv2.circle(temp_img, (ccx, ccy), cr, (0, 255, 0), 2)
            texts = ["程序已自动识别观察仓(绿圈)", "空格=确认  R=手动重画  ESC=退出"]
        else:
            if rect[2] > 0 and rect[3] > 0:
                cv2.rectangle(temp_img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)
                cx, cy = rect[0]+rect[2]//2, rect[1]+rect[3]//2
                r = min(rect[2], rect[3])//2
                cv2.circle(temp_img, (cx, cy), r, (0, 255, 255), 2)
            texts = ["请用鼠标框选圆形观察仓", "空格=确认  ESC=退出"]
        temp_img = put_chinese_text(temp_img, texts, (20, 20), (0, 255, 0), 28)
        cv2.imshow(win_name_1, temp_img)
        key = cv2.waitKey(30) & 0xFF
        if key in [13, 10, 32]:
            if not mode_1_manual and auto_circle is not None:
                circle_params = (int(auto_circle[0]/scale), int(auto_circle[1]/scale), int(auto_circle[2]/scale)); break
            elif mode_1_manual and rect[2] > 10 and rect[3] > 10:
                cx, cy = rect[0]+rect[2]//2, rect[1]+rect[3]//2
                r = min(rect[2], rect[3])//2
                circle_params = (int(cx/scale), int(cy/scale), int(r/scale)); break
        elif key in [ord('r'), ord('R')]:
            mode_1_manual = True; auto_circle = None; rect = (0, 0, 0, 0)
        elif key == 27:
            cv2.destroyAllWindows(); cap.release(); return None
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    safe_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(safe_mask, circle_params[:2], max(10, circle_params[2] - 5), 255, -1)
    print("计算时序中值背景...")
    bg_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in cached_frames]
    bg = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)

    # ===== 步骤2：学习虫体 =====
    params = {
        "dark_worm": True,
        "target_area": None, "target_length": None, "target_aspect": None,
        "min_area": 30, "max_area": 100000, "min_length": 10,
        "circle_params": circle_params, "bg": bg, "safe_mask": safe_mask,
    }
    win_name_2 = "Step 2: 学习虫体特征"
    cv2.namedWindow(win_name_2, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win_name_2, mouse_draw_rect)
    frame_idx = 0
    error_msg = ""
    manual_mode = False
    while True:
        if is_window_closed(win_name_2): cap.release(); return None
        orig_frame = cached_frames[frame_idx % len(cached_frames)]
        frame_disp_2 = cv2.resize(orig_frame, (int(w*scale), int(h*scale))) if scale < 1.0 else orig_frame.copy()
        temp_img = frame_disp_2.copy()
        disp_ccx, disp_ccy = int(circle_params[0]*scale), int(circle_params[1]*scale)
        cv2.circle(temp_img, (disp_ccx, disp_ccy), int(circle_params[2]*scale), (200, 200, 200), 1)
        current_area, current_aspect, current_length = 0, 0, 0
        if not manual_mode:
            g_allow_draw = False
            gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
            temp_params = params.copy()
            if params["target_area"] is None:
                temp_params["min_area"] = 30
                temp_params["max_area"] = 100000
                temp_params["min_length"] = 5
            candidates = detect_candidates(gray, bg, safe_mask, temp_params)
            if candidates:
                best = max(candidates, key=lambda c: c["area"])
                bx, by, bw, bh = best["bbox"]
                cv2.rectangle(temp_img, (int(bx*scale), int(by*scale)),
                              (int((bx+bw)*scale), int((by+bh)*scale)), (0, 255, 0), 2)
                current_area = best["area"]
                current_length = best["length"]
                current_aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                error_msg = ""
            else:
                error_msg = "本图未自动找到目标"
            texts = [
                "【学习虫体】绿框=程序识别结果",
                "1=跳过本帧  2=认可并学习",
                "3=手动框选(强迫学习)  4=学够了,开跑!",
                error_msg
            ]
        else:
            g_allow_draw = True
            if rect[2] > 0:
                cv2.rectangle(temp_img, (rect[0], rect[1]),
                              (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)
            texts = ["【手动模式】框选完整虫体", "C=确认  R=取消"]
        color = (0, 0, 255) if error_msg else (0, 255, 255)
        temp_img = put_chinese_text(temp_img, texts, (20, 20), color, 26)
        cv2.imshow(win_name_2, temp_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('1'):
            frame_idx += 1; manual_mode = False; rect = (0, 0, 0, 0); error_msg = ""
        elif key == ord('2'):
            if not manual_mode and current_area > 0:
                params["target_area"] = current_area
                params["target_length"] = current_length
                params["target_aspect"] = current_aspect
                params["min_area"] = max(20, current_area * 0.25)
                params["max_area"] = current_area * 4.0
                params["min_length"] = max(8, current_length * 0.4)
                print(f"[学习] area={current_area:.0f} len={current_length} asp={current_aspect:.2f} dark={params['dark_worm']}")
                frame_idx += 1; manual_mode = False; rect = (0, 0, 0, 0); error_msg = ""
        elif key == ord('3'):
            manual_mode = True; rect = (0, 0, 0, 0)
        elif key == ord('4'):
            if params["target_area"] is None:
                error_msg = "请先按3框选一次虫子!"
            else:
                print("✅ 学习完成!")
                cv2.destroyAllWindows()
                for _ in range(5): cv2.waitKey(1)
                cap.release(); return params
        elif key in [ord('r'), ord('R')]:
            if manual_mode:
                manual_mode = False; rect = (0, 0, 0, 0)
        elif key in [ord('c'), ord('C')]:
            if manual_mode and rect[2] > 10 and rect[3] > 10:
                hx, hy, hw, hh = [int(v/scale) for v in rect]
                roi_gray = cv2.cvtColor(orig_frame[hy:hy+hh, hx:hx+hw], cv2.COLOR_BGR2GRAY)
                if roi_gray.size > 0:
                    if roi_gray.mean() - roi_gray.min() > roi_gray.max() - roi_gray.mean():
                        params["dark_worm"] = True
                    else:
                        params["dark_worm"] = False
                    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                    thresh_type = cv2.THRESH_BINARY_INV if params["dark_worm"] else cv2.THRESH_BINARY
                    _, local_mask = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        best_cnt = max(cnts, key=cv2.contourArea)
                        area = cv2.contourArea(best_cnt)
                        if area < 20 or area > roi_gray.size * 0.8:
                            error_msg = "提取面积异常,请重选"
                        else:
                            bx, by, bw, bh = cv2.boundingRect(best_cnt)
                            length = max(bw, bh)
                            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                            params["target_area"] = area
                            params["target_length"] = length
                            params["target_aspect"] = aspect
                            params["min_area"] = max(20, area * 0.25)
                            params["max_area"] = area * 4.0
                            params["min_length"] = max(8, length * 0.4)
                            print(f"[手动学习] dark={params['dark_worm']} area={area:.0f} len={length} asp={aspect:.2f}")
                            frame_idx += 1; manual_mode = False; rect = (0, 0, 0, 0); error_msg = ""
        elif key == 27:
            cv2.destroyAllWindows(); cap.release(); return None

# ================= 批量处理 =================
def process_one_video(video_path, output_root, params):
    video_path = Path(video_path)
    vid_name = video_path.stem
    out_dir = Path(output_root) / vid_name
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "worm_crops"
    crop_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total <= 0:
        cap.release(); return {"video": vid_name, "error": "元数据读取失败"}
    if total < fps * TRIM_SECONDS * 2 + 10:
        start_f, end_f = 0, total
    else:
        start_f = int(fps * TRIM_SECONDS)
        end_f = int(total - fps * TRIM_SECONDS)

    print(f"[{vid_name}] 估计背景...")
    sample_idxs = np.linspace(start_f, max(start_f+1, end_f-1), 20).astype(int)
    bg_frames = []
    for i in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret: bg_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    bg = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8) if bg_frames else None
    if bg is not None:
        # 保存背景图，供 Step2 做背景差（去除舱壁/静态杂物）
        cv2.imwrite(str(out_dir / "background.png"), bg)

    safe_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(safe_mask, params["circle_params"][:2], max(10, params["circle_params"][2] - 5), 255, -1)

    step = max(1, int(round(fps / TARGET_FPS)))
    frame_indices = list(range(start_f, end_f, step))

    tracker = Tracker()
    records = []
    crop_size = int(max(params["target_length"] * CROP_SCALE, 64))
    search_r = params["target_length"] * 1.5

    print(f"\n[{vid_name}] 开始处理，共 {len(frame_indices)} 帧")
    progress = SingleLineProgress(
        total=len(frame_indices),
        prefix=f"[{vid_name[:16]:<16}] ",
        bar_width=28,
    )

    for seq_idx, fidx in enumerate(frame_indices):
        progress.update(seq_idx + 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred = tracker.predict()
        if pred is not None and tracker.miss < 5:
            roi = clip_roi((pred[0]-search_r, pred[1]-search_r, pred[0]+search_r, pred[1]+search_r), w, h)
            candidates = detect_candidates(gray, bg, safe_mask, params, roi=roi)
            if not candidates:
                candidates = detect_candidates(gray, bg, safe_mask, params, roi=None)
        else:
            candidates = detect_candidates(gray, bg, safe_mask, params, roi=None)

        if not candidates:
            tracker.mark_miss(); continue

        best = max(candidates, key=lambda c: score_candidate(c, pred, params))
        if score_candidate(best, pred, params) <= 0:
            tracker.mark_miss(); continue

        cx, cy = best["centroid"]
        tracker.update((cx, cy))

        x1 = max(0, int(cx - crop_size//2)); y1 = max(0, int(cy - crop_size//2))
        x2 = min(w, x1 + crop_size); y2 = min(h, y1 + crop_size)
        if x2 - x1 < crop_size: x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size: y1 = max(0, y2 - crop_size)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        if CROP_SAVE_AS_JPEG:
            crop_name = f"worm_{seq_idx:06d}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
        else:
            crop_name = f"worm_{seq_idx:06d}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
        records.append({
            "frame_index": fidx, "seq_index": seq_idx,
            "center_x": float(cx), "center_y": float(cy),
            "area": float(best["area"]), "length": float(best["length"]),
            "crop_x": float(x1), "crop_y": float(y1),
            "crop_file": crop_name,
        })

    progress.done()
    cap.release()

    if records:
        df = pd.DataFrame(records)
        df = df.sort_values("frame_index").reset_index(drop=True)
        df.to_csv(out_dir / "worm_trajectory.csv", index=False)
        if len(df) > 1:
            # 速度用真实时间差：dt = Δframe_index / fps，跳帧处不会被当成恒定 1/TARGET_FPS
            dx = df["center_x"].diff(); dy = df["center_y"].diff()
            dt_sec = np.maximum(df["frame_index"].diff().values / fps, 1e-6)
            speed = np.sqrt(dx**2 + dy**2) / dt_sec
            duration_s = float((df["frame_index"].iloc[-1] - df["frame_index"].iloc[0]) / fps)
            pd.DataFrame([{
                "video": vid_name, "n_frames": len(df),
                "duration_s": duration_s,
                "fps": fps, "target_fps": TARGET_FPS,
                "chamber_center_x": float(params["circle_params"][0]),
                "chamber_center_y": float(params["circle_params"][1]),
                "chamber_radius_px": float(params["circle_params"][2]),
                "mean_speed_px_s": float(np.nanmean(speed)),
                "max_speed_px_s": float(np.nanmax(speed)),
            }]).to_csv(out_dir / "worm_stats.csv", index=False)
        return {"video": vid_name, "n_valid_frames": len(records)}
    else:
        return {"video": vid_name, "error": "未生成有效记录"}

# ================= 主程序 =================
def main():
    from ConfigWindow import ask_config

    script_dir = Path(__file__).resolve().parent
    cfg = ask_config(default_input_dir=script_dir, default_output_dir=None,
                     title="Step1 - 检测与裁剪参数",
                     show_video_list=True, show_scale=False, show_bg=False,
                     show_plot=False, show_dump=False, show_fps=False)
    if cfg is None:
        return

    video_files = [Path(p) for p in cfg.get("files", []) if Path(p).exists()]
    if not video_files:
        print("未勾选任何视频。")
        return
    output_root = Path(cfg["output"]) if cfg["output"] else Path(cfg["input"]) / "step1_output"

    params = interactive_learning(video_files[0])
    if params is None: return

    print(f"\n共 {len(video_files)} 个视频待处理")
    ok, fail = [], []
    for v in video_files:
        res = process_one_video(v, output_root, params)
        if "error" in res:
            fail.append(res); print(f"❌ {v.name}: {res['error']}")
        else:
            ok.append(res); print(f"✅ {v.name}: {res['n_valid_frames']} 帧")

    print(f"\n🎉 {len(ok)}成功 {len(fail)}失败")
    if ok:
        pd.DataFrame(ok).to_csv(output_root / "batch_summary.csv", index=False)

if __name__ == "__main__":
    main()