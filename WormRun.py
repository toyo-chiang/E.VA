# -*- coding: utf-8 -*-
"""
WormRun.py —— 一步完成：交互学习 → 逐帧检测 + 体形分析 → 图表与汇总。

设计要点：
- 不落盘裁剪图（默认），省磁盘、省 I/O；需要人工核对时加 --dump-crops。
- 保留 background.png（背景图）与 worm_analysis.png（图表）。
- 复用 Step1 的检测/跟踪与 Step2 的体形分析，逻辑与两步版一致。

用法（命令行）：
    python WormRun.py --input 视频目录 --output step1_output --um-per-px 2.0 --plot
    python WormRun.py --input 单个视频.mp4 --output out
    python WormRun.py --dump-crops          # 额外保存裁剪图，便于排查/复用

用法（窗口）：直接 python WormRun.py（不带 --input）
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from Step1 import (
    interactive_learning, detect_candidates, score_candidate, Tracker, clip_roi,
    CROP_SCALE, TARGET_FPS, TRIM_SECONDS, CROP_SAVE_AS_JPEG, CROP_JPEG_QUALITY,
    SingleLineProgress,
)
from Step2 import analyze_crop, estimate_frequency, smooth_ma, _plot, batch_filename

warnings.filterwarnings("ignore")


# ============================================================ 单视频：检测 + 体形分析 ============================================================
def process_video(video_path, output_root, params, args):
    video_path = Path(video_path)
    vid_name = video_path.stem
    out_dir = Path(output_root) / vid_name
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_dir = None
    if args.dump_crops:
        crop_dir = out_dir / "worm_crops"
        crop_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if getattr(args, "fps", None):
        fps = float(args.fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total <= 0:
        cap.release()
        return {"video": vid_name, "error": "元数据读取失败"}

    if total < fps * TRIM_SECONDS * 2 + 10:
        start_f, end_f = 0, total
    else:
        start_f = int(fps * TRIM_SECONDS)
        end_f = int(total - fps * TRIM_SECONDS)

    print(f"[{vid_name}] 估计背景...")
    sample_idxs = np.linspace(start_f, max(start_f + 1, end_f - 1), 20).astype(int)
    bg_frames = []
    for i in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            bg_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    bg = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8) if bg_frames else None
    if bg is not None:
        cv2.imwrite(str(out_dir / "background.png"), bg)

    safe_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(safe_mask, params["circle_params"][:2], max(10, params["circle_params"][2] - 5), 255, -1)

    step = max(1, int(round(fps / TARGET_FPS)))
    frame_indices = list(range(start_f, end_f, step))
    tracker = Tracker()
    crop_size = int(max(params["target_length"] * CROP_SCALE, 64))
    search_r = params["target_length"] * 1.5

    rows = []
    n_ok = 0
    n_skip = 0
    skip_reasons = {}
    traj_records = []
    progress = SingleLineProgress(len(frame_indices), prefix=f"[{vid_name[:16]:<16}] ")

    for seq_idx, fidx in enumerate(frame_indices):
        progress.update(seq_idx + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred = tracker.predict()
        if pred is not None and tracker.miss < 5:
            roi = clip_roi((pred[0] - search_r, pred[1] - search_r,
                            pred[0] + search_r, pred[1] + search_r), w, h)
            candidates = detect_candidates(gray, bg, safe_mask, params, roi=roi)
            if not candidates:
                candidates = detect_candidates(gray, bg, safe_mask, params, roi=None)
        else:
            candidates = detect_candidates(gray, bg, safe_mask, params, roi=None)

        row = {
            "frame_index": int(fidx), "seq_index": int(seq_idx),
            "centroid_x": np.nan, "centroid_y": np.nan,
            "head_x": np.nan, "head_y": np.nan,
            "tail_x": np.nan, "tail_y": np.nan,
            "length_px": np.nan, "width_px": np.nan,
            "bend_head_tail_deg": np.nan, "curv_mid_deg": np.nan,
            "polarity": None,
        }

        if not candidates:
            n_skip += 1
            skip_reasons["无候选"] = skip_reasons.get("无候选", 0) + 1
            rows.append(row)
            continue

        best = max(candidates, key=lambda c: score_candidate(c, pred, params))
        if score_candidate(best, pred, params) <= 0:
            n_skip += 1
            skip_reasons["得分过低"] = skip_reasons.get("得分过低", 0) + 1
            rows.append(row)
            continue

        cx, cy = best["centroid"]
        tracker.update((cx, cy))
        row["centroid_x"] = float(cx)
        row["centroid_y"] = float(cy)

        x1 = max(0, int(cx - crop_size // 2)); y1 = max(0, int(cy - crop_size // 2))
        x2 = min(w, x1 + crop_size); y2 = min(h, y1 + crop_size)
        if x2 - x1 < crop_size: x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size: y1 = max(0, y2 - crop_size)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            n_skip += 1
            skip_reasons["空裁剪"] = skip_reasons.get("空裁剪", 0) + 1
            rows.append(row)
            continue

        if crop_dir is not None:
            if CROP_SAVE_AS_JPEG:
                cname = f"worm_{seq_idx:06d}.jpg"
                cv2.imwrite(str(crop_dir / cname), crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
            else:
                cname = f"worm_{seq_idx:06d}.png"
                cv2.imwrite(str(crop_dir / cname), crop)
            traj_records.append({
                "frame_index": int(fidx), "seq_index": int(seq_idx),
                "center_x": float(cx), "center_y": float(cy),
                "area": float(best["area"]), "length": float(best["length"]),
                "crop_x": float(x1), "crop_y": float(y1), "crop_file": cname,
            })

        bg_crop = bg[y1:y2, x1:x2] if (bg is not None and not args.no_bg) else None
        res, reason = analyze_crop(crop, bg_crop)
        if res is None:
            n_skip += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            rows.append(row)
            continue
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

    progress.done()
    cap.release()

    if crop_dir is not None and traj_records:
        # 同时写一份 Step2 兼容的轨迹 + 统计 CSV，便于之后单跑 Step2 复核
        pd.DataFrame(traj_records).to_csv(out_dir / "worm_trajectory.csv", index=False)
        pd.DataFrame([{
            "video": vid_name, "fps": fps, "target_fps": TARGET_FPS,
            "chamber_center_x": float(params["circle_params"][0]),
            "chamber_center_y": float(params["circle_params"][1]),
            "chamber_radius_px": float(params["circle_params"][2]),
        }]).to_csv(out_dir / "worm_stats.csv", index=False)

    return _finalize(out_dir, rows, fps, args.um_per_px, args.plot,
                     params["circle_params"], n_ok, n_skip, skip_reasons)


# ============================================================ 汇总/速度/频率/图表 ============================================================
def _finalize(folder, rows, fps, um_per_px, do_plot, circle, n_ok, n_skip, skip_reasons):
    if not rows:
        return {"folder": folder.name, "error": "没有可分析的帧"}

    f = pd.DataFrame(rows)
    f = f.sort_values("frame_index").reset_index(drop=True)

    # 头尾时序一致性（跳过的帧 head 为 NaN，跳过）
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

    t = f["frame_index"].values / fps

    # 轨迹（质心，轻度平滑，保持原样）
    sm_x = smooth_ma(f["centroid_x"].values, 5)
    sm_y = smooth_ma(f["centroid_y"].values, 5)

    # 速度：先对质心中值滤波去孤立跳变，再按真实 dt 求速度
    vx = pd.Series(f["centroid_x"].values).rolling(5, center=True, min_periods=1).median().values
    vy = pd.Series(f["centroid_y"].values).rolling(5, center=True, min_periods=1).median().values
    dx = np.diff(vx); dy = np.diff(vy)
    d_frame = np.diff(f["frame_index"].values)
    dt_sec = np.maximum(d_frame / fps, 1e-6)
    speed_px = np.concatenate([[np.nan], np.sqrt(dx ** 2 + dy ** 2) / dt_sec])
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

    curv_all = f["curv_mid_deg"].values
    bend_all = f["bend_head_tail_deg"].values
    valid = ~np.isnan(curv_all)
    curv = curv_all[valid]
    bend = bend_all[valid]
    t_valid = t[valid]
    f_curv = estimate_frequency(t_valid, curv)
    f_bend = estimate_frequency(t_valid, bend)

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


# ============================================================ 批量入口 ============================================================
def _resolve_videos(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path], input_path.parent / "step1_output"
    vids = sorted([p for p in input_path.glob("*.mp4")] + [p for p in input_path.glob("*.avi")])
    return vids, input_path / "step1_output"


def run(video_files, output_root, fps, um_per_px, do_plot, no_bg, dump_crops, batch_name=None):
    video_files = [Path(v) for v in video_files if Path(v).exists()]
    if not video_files:
        print("没有可处理的视频。")
        return
    output_root = Path(output_root)

    # 交互学习只在第一个视频上做一次
    params = interactive_learning(video_files[0])
    if params is None:
        print("交互学习被取消，退出。")
        return

    args = argparse.Namespace(um_per_px=um_per_px, plot=do_plot,
                              no_bg=no_bg, dump_crops=dump_crops, fps=fps)
    summaries = []
    for v in video_files:
        res = process_video(v, output_root, params, args)
        if "error" in res:
            print(f"❌ {v.name}: {res['error']}")
        else:
            summaries.append(res)
            print(f"✅ {v.name}: 分析 {res['n_analyzed']} 帧 / 跳过 {res['n_skipped']} 帧 "
                  f"(覆盖率 {res['coverage']*100:.1f}%)")

    if summaries:
        out2 = output_root.parent / "step2_output"
        out2.mkdir(parents=True, exist_ok=True)
        batch = out2 / batch_filename(batch_name)
        pd.DataFrame(summaries).to_csv(batch, index=False)
        print(f"\n批量汇总: {batch}")


# ============================================================ 窗口模式 ============================================================
def run_gui():
    from tkinter import messagebox
    from ConfigWindow import ask_config

    script_dir = Path(__file__).resolve().parent
    cfg = ask_config(default_input_dir=script_dir, default_output_dir=None,
                     title="线虫分析 - 参数设置",
                     show_video_list=True, show_scale=True, show_bg=True,
                     show_plot=True, show_dump=True, show_fps=True, show_batch_name=True)
    if cfg is None:
        return
    files = cfg.get("files", [])
    if not files:
        messagebox.showwarning("提示", "请至少勾选一个视频。")
        return
    output_root = Path(cfg["output"]) if cfg["output"] else Path(cfg["input"]) / "step1_output"
    run(files, output_root, cfg["fps"], cfg["um_per_px"], cfg["plot"],
        not cfg["use_bg"], cfg["dump_crops"], batch_name=cfg.get("batch_name"))
    messagebox.showinfo("完成", "处理结束，请查看输出目录。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="视频目录或单个视频；不填则弹窗")
    ap.add_argument("--output", default=None, help="输出根目录；默认 输入目录/step1_output")
    ap.add_argument("--fps", type=float, default=None, help="原始视频帧率（保留参数，当前自动从视频读取）")
    ap.add_argument("--um-per-px", type=float, default=None, help="像素->微米标尺")
    ap.add_argument("--plot", action="store_true", help="输出图表")
    ap.add_argument("--no-bg", action="store_true", help="禁用背景差")
    ap.add_argument("--dump-crops", action="store_true", help="额外保存裁剪图（排查/复用）")
    ap.add_argument("--batch-name", default=None, help="批次汇总 CSV 名称；留空按时间命名")
    args = ap.parse_args()

    if args.input:
        video_files, default_out = _resolve_videos(args.input)
        output_root = Path(args.output) if args.output else default_out
        run(video_files, output_root, args.fps, args.um_per_px, args.plot,
            args.no_bg, args.dump_crops, batch_name=args.batch_name)
    else:
        run_gui()


if __name__ == "__main__":
    main()
