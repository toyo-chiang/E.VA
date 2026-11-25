# -*- coding: utf-8 -*-
"""
ConfigWindow.py —— 统一的参数设置窗口。

两种形态：
1) show_video_list=True：先选文件夹，右侧列出该文件夹里的视频（可滚动、可勾选，
   配"全选/全不选"），勾选后点"确定"。适合"同一文件夹里既有要分析的、也有要保留的异常样本"。
2) show_video_list=False：普通的输入目录 + 分析选项（给 Step2 这类处理输出目录的脚本用）。

对外只暴露 ask_config(...)，返回配置 dict；取消返回 None。
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

VIDEO_EXTS = (".mp4", ".avi")


def _list_videos(folder):
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VIDEO_EXTS)


def ask_config(default_input_dir=None, default_output_dir=None, title="参数设置",
               show_video_list=False, show_scale=True, show_bg=True, show_plot=True,
               show_dump=False, show_fps=False, show_batch_name=False):
    result = {"cancel": True, "files": []}

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    pad = {"padx": 8, "pady": 4}

    # ---------------- 变量 ----------------
    v_input = tk.StringVar(value=str(default_input_dir or ""))
    v_output = tk.StringVar(value=str(default_output_dir or ""))
    v_scale_on = tk.BooleanVar(value=False)
    v_scale = tk.StringVar(value="")
    v_bg = tk.BooleanVar(value=True)
    v_plot = tk.BooleanVar(value=True)
    v_dump = tk.BooleanVar(value=False)
    v_fps = tk.StringVar(value="")
    v_batch = tk.StringVar(value="")

    # ---------------- 输入行 ----------------
    f_in = tk.Frame(root)
    f_in.pack(fill="x", **pad)
    if show_video_list:
        tk.Label(f_in, text="视频文件夹：", width=14, anchor="e").pack(side="left")
        tk.Entry(f_in, textvariable=v_input, width=42).pack(side="left", padx=(0, 4))

        def _browse_dir():
            p = filedialog.askdirectory(title="选择视频文件夹")
            if p:
                v_input.set(p)
                _refresh()

        tk.Button(f_in, text="浏览…", command=_browse_dir).pack(side="left")
        tk.Button(f_in, text="刷新列表", command=lambda: _refresh()).pack(side="left", padx=(4, 0))
    else:
        tk.Label(f_in, text="输入目录：", width=14, anchor="e").pack(side="left")
        tk.Entry(f_in, textvariable=v_input, width=42).pack(side="left", padx=(0, 4))

        def _browse_input():
            p = filedialog.askdirectory(title="选择输入目录")
            if p:
                v_input.set(p)

        tk.Button(f_in, text="浏览…", command=_browse_input).pack(side="left")

    # ---------------- 输出行 ----------------
    f_out = tk.Frame(root)
    f_out.pack(fill="x", **pad)
    tk.Label(f_out, text="输出目录：", width=14, anchor="e").pack(side="left")
    tk.Entry(f_out, textvariable=v_output, width=42).pack(side="left", padx=(0, 4))

    def _browse_out():
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            v_output.set(p)

    tk.Button(f_out, text="浏览…", command=_browse_out).pack(side="left")

    # ---------------- 批次名称（可选） ----------------
    if show_batch_name:
        f_batch = tk.Frame(root)
        f_batch.pack(fill="x", **pad)
        tk.Label(f_batch, text="批次名称：", width=14, anchor="e").pack(side="left")
        tk.Entry(f_batch, textvariable=v_batch, width=30).pack(side="left")
        tk.Label(f_batch, text="（留空=按时间命名）", fg="gray").pack(side="left", padx=(4, 0))

    # ---------------- 主体：左设置 / 右视频列表 ----------------
    body = tk.Frame(root)
    body.pack(fill="both", expand=True, **pad)
    left = tk.Frame(body)
    left.pack(side="left", fill="y", padx=(0, 12))
    right = tk.Frame(body)
    right.pack(side="left", fill="both", expand=True)

    # ---- 左：分析选项 ----
    if show_scale:
        f_scale = tk.Frame(left)
        f_scale.pack(fill="x", pady=4)
        cb_scale = tk.Checkbutton(f_scale, text="使用像素标尺 (µm/px)：", variable=v_scale_on)
        cb_scale.pack(side="left")
        ent_scale = tk.Entry(f_scale, textvariable=v_scale, width=10)
        ent_scale.pack(side="left")

        def _toggle_scale():
            ent_scale.config(state="normal" if v_scale_on.get() else "disabled")

        cb_scale.config(command=_toggle_scale)
        _toggle_scale()

    if show_bg:
        tk.Checkbutton(left, text="使用背景差去除舱壁/静态杂物（推荐）", variable=v_bg).pack(anchor="w", pady=4)
    if show_plot:
        tk.Checkbutton(left, text="生成图表（轨迹/速度/弯曲/频谱等）", variable=v_plot).pack(anchor="w", pady=4)
    if show_dump:
        tk.Checkbutton(left, text="额外保存裁剪图（排查/复用，占用空间）", variable=v_dump).pack(anchor="w", pady=4)

    if show_fps:
        f_fps = tk.Frame(left)
        f_fps.pack(fill="x", pady=4)
        tk.Label(f_fps, text="原始帧率 fps（留空自动）：").pack(side="left")
        tk.Entry(f_fps, textvariable=v_fps, width=8).pack(side="left")

    # ---- 右：视频列表（仅 show_video_list） ----
    video_vars = {}
    inner = None
    last_refreshed = [None]

    if show_video_list:
        hdr = tk.Frame(right)
        hdr.pack(fill="x", pady=4)
        tk.Label(hdr, text="视频列表（勾选要分析的）：").pack(side="left")
        tk.Button(hdr, text="全选", command=lambda: _set_all(True)).pack(side="right")
        tk.Button(hdr, text="全不选", command=lambda: _set_all(False)).pack(side="right", padx=(0, 4))

        # 带边框的列表容器（sunken 下凹边框，让区域边界清楚）
        lst = tk.Frame(right, relief="sunken", bd=1, bg="#ffffff")
        lst.pack(fill="both", expand=True)
        canvas = tk.Canvas(lst, width=340, height=250, highlightthickness=0, bg="#ffffff")
        sb = tk.Scrollbar(lst, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#ffffff")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 鼠标滚轮：Canvas 默认不响应，手动绑定到 canvas / inner / 每个子控件
        def _on_wheel(event):
            d = getattr(event, "delta", 0)
            if d == 0:
                return
            steps = int(-d / 120) if abs(d) >= 120 else int(-d)
            canvas.yview_scroll(steps, "units")

        def _on_wheel_linux(event, step):
            canvas.yview_scroll(step, "units")

        def _bind_wheel(widget):
            widget.bind("<MouseWheel>", _on_wheel)
            widget.bind("<Button-4>", lambda e: _on_wheel_linux(e, -1))
            widget.bind("<Button-5>", lambda e: _on_wheel_linux(e, 1))

        _bind_wheel(canvas)
        _bind_wheel(inner)

        def _clear():
            for w in inner.winfo_children():
                w.destroy()
            video_vars.clear()

        def _refresh():
            _clear()
            folder = v_input.get().strip()
            last_refreshed[0] = folder
            for p in _list_videos(folder):
                var = tk.BooleanVar(value=True)   # 默认全选
                cb = tk.Checkbutton(inner, text=p.name, variable=var, anchor="w", bg="#ffffff")
                cb.pack(fill="x", anchor="w")
                _bind_wheel(cb)
                video_vars[str(p)] = var
            canvas.yview_moveto(0)

        def _set_all(flag):
            for var in video_vars.values():
                var.set(flag)
    else:
        def _refresh():
            pass

        def _set_all(flag):
            pass

    # ---------------- 确定 / 取消 ----------------
    def on_ok():
        inp = v_input.get().strip()
        if not inp:
            messagebox.showerror("提示", "请填写输入目录。")
            return
        p = Path(inp)
        if not p.exists():
            messagebox.showerror("提示", f"路径不存在：\n{inp}")
            return

        files = []
        if show_video_list:
            if not p.is_dir():
                messagebox.showerror("提示", "请选择一个视频文件夹。")
                return
            if v_input.get().strip() != last_refreshed[0]:
                _refresh()   # 文件夹改了但没刷新：先加载，再读勾选
            files = [fp for fp, var in video_vars.items() if var.get()]
            if not files:
                messagebox.showerror("提示", "请至少勾选一个视频。")
                return

        scale = None
        if v_scale_on.get():
            try:
                scale = float(v_scale.get().strip())
                if scale <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("提示", "像素标尺必须是正数（µm/px）。")
                return

        fps = None
        if show_fps and v_fps.get().strip():
            try:
                fps = float(v_fps.get().strip())
                if fps <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("提示", "帧率必须是正数。")
                return

        result.update(
            cancel=False, input=inp, files=files,
            output=(v_output.get().strip() or None),
            um_per_px=scale, use_bg=v_bg.get(), plot=v_plot.get(),
            dump_crops=v_dump.get(), fps=fps, batch_name=v_batch.get().strip(),
        )
        root.destroy()

    f_btn = tk.Frame(root)
    f_btn.pack(fill="x", pady=10)
    tk.Button(f_btn, text="确定", width=10, command=on_ok).pack(side="right", padx=8)
    tk.Button(f_btn, text="取消", width=10, command=lambda: root.destroy()).pack(side="right", padx=8)

    if show_video_list:
        _refresh()

    root.mainloop()
    return None if result.get("cancel", True) else result
