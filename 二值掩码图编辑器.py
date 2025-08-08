# mask_editor_full.py
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2

# ----------------------------
# 配置与常量
# ----------------------------
RESOLUTIONS = {
    "VGA (640x480)": (640, 480),
    "QVGA (320x240)": (320, 240),
    "QQVGA (160x120)": (160, 120),
    "QQQVGA (80x60)": (80, 60)
}

DEFAULT_LAB = (200, 0, 150, 100, 150, 100)  # Lmax,Lmin,Amax,Amin,Bmax,Bmin
DEFAULT_GRAY_BIN = (0, 128)  # min,max for gray bin (when importing small it's fine)


# ----------------------------
# 工具函数
# ----------------------------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def ensure_binary_np(arr, thresh=128):
    return np.where(arr > thresh, 255, 0).astype(np.uint8)


# ----------------------------
# 主类
# ----------------------------
class MaskEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("二值掩码图编辑器")
        self.root.geometry("1100x700")
        style = ttk.Style(root)
        style.theme_use("clam")

        # 状态
        self.target_resolution = RESOLUTIONS["VGA (640x480)"]
        self.image = None              # PIL Image 当前编辑内容（L 模式）
        self.original_image = None     # 原始进入编辑区的图像（用于重置/擦除）
        self.tk_img = None
        self.canvas_image_id = None
        self.border_id = None

        # view transform state
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None

        # editing
        self.tool = "paint"  # "paint" means draw black rect, "erase" means restore white
        self.drag_start = None
        self.preview_snapshot = None

        # history
        self.undo_stack = []
        self.redo_stack = []

        # UI 布局
        self._build_ui()

        # 绑定键
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("z", lambda e: self.undo())
        self.root.bind("y", lambda e: self.redo())

    # ----------------------------
    # UI 构建
    # ----------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧工具栏
        toolbar = ttk.Frame(main, width=260)
        toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        ttk.Label(toolbar, text="工具栏", font=("Segoe UI", 14, "bold")).pack(pady=6)

        # 分辨率
        ttk.Label(toolbar, text="分辨率：").pack(anchor="w", pady=(8, 0))
        self.res_var = tk.StringVar(value="VGA (640x480)")
        res_combo = ttk.Combobox(toolbar, textvariable=self.res_var, values=list(RESOLUTIONS.keys()), state="readonly")
        res_combo.pack(fill=tk.X, pady=3)

        ttk.Button(toolbar, text="生成白板", command=self.generate_white).pack(fill=tk.X, pady=6)
        ttk.Button(toolbar, text="导入图片", command=self.import_image_dialog).pack(fill=tk.X, pady=6)

        # 导入选项说明区（阈值、模式）
        ttk.Separator(toolbar).pack(fill=tk.X, pady=6)
        ttk.Label(toolbar, text="导入处理：").pack(anchor="w", pady=(4, 2))
        self.import_mode_var = tk.StringVar(value="灰度化")
        frame_modes = ttk.Frame(toolbar)
        frame_modes.pack(fill=tk.X)
        ttk.Radiobutton(frame_modes, text="灰度化", variable=self.import_mode_var, value="灰度化").pack(side=tk.LEFT)
        ttk.Radiobutton(frame_modes, text="二值化", variable=self.import_mode_var, value="二值化").pack(side=tk.LEFT)

        ttk.Label(toolbar, text="阈值输入：").pack(anchor="w", pady=(8, 2))
        self.threshold_entry = ttk.Entry(toolbar)
        self.threshold_entry.pack(fill=tk.X)
        self.threshold_entry.insert(0, "默认：LAB 或 灰度区间")

        # 编辑工具
        ttk.Separator(toolbar).pack(fill=tk.X, pady=8)
        ttk.Label(toolbar, text="编辑工具：").pack(anchor="w", pady=(4, 2))
        btn_paint = ttk.Button(toolbar, text="画黑（矩形）", command=lambda: self.set_tool("paint"))
        btn_erase = ttk.Button(toolbar, text="擦除（矩形）", command=lambda: self.set_tool("erase"))
        btn_paint.pack(fill=tk.X, pady=3)
        btn_erase.pack(fill=tk.X, pady=3)

        # 网格、重置、撤销重做、保存
        self.grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="显示像素网格", variable=self.grid_var, command=self.redraw_canvas).pack(anchor="w", pady=6)

        ttk.Button(toolbar, text="重置为原始", command=self.reset_to_original).pack(fill=tk.X, pady=4)
        ttk.Button(toolbar, text="撤销 (Z)", command=self.undo).pack(fill=tk.X, pady=4)
        ttk.Button(toolbar, text="重做 (Y)", command=self.redo).pack(fill=tk.X, pady=4)

        ttk.Button(toolbar, text="保存掩码", command=self.save_mask).pack(fill=tk.X, pady=(12, 4))
        ttk.Label(toolbar, text="说明：拖拽框选编辑；滚轮缩放；中键拖动平移").pack(pady=(6, 0))

        # 右侧画布区
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # canvas 放在带背景的框中，使得画布居中更明显
        self.canvas_bg = tk.Canvas(right, bg="#bdbdbd")  # 灰色背景，突出白板边缘
        self.canvas_bg.pack(fill=tk.BOTH, expand=True)
        # 主画布放置在 canvas_bg 上（通过 create_window），便于居中
        self.canvas = tk.Canvas(self.canvas_bg, bg="#999999", highlightthickness=0)
        self.canvas_id_window = self.canvas_bg.create_window(10, 10, anchor="nw", window=self.canvas)

        # 绑定事件（缩放、平移、绘制）
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_down)  # 中键开始平移
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows / Mac
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down

        # 当窗口大小变化时更新居中显示
        self.canvas_bg.bind("<Configure>", lambda e: self.redraw_canvas())

        # 状态栏底部
        self.status_var = tk.StringVar(value="准备中")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化画布为空白提示
        self.show_placeholder()

    # ----------------------------
    # 操作函数：生成、导入、阈值解析
    # ----------------------------
    def generate_white(self):
        res = self.res_var.get()
        if res not in RESOLUTIONS:
            messagebox.showerror("错误", "请选择分辨率后再生成")
            return
        self.target_resolution = RESOLUTIONS[res]
        w, h = self.target_resolution
        self.image = Image.new("L", (w, h), 255)
        self.original_image = self.image.copy()
        self.push_history()
        self.reset_view()
        self.redraw_canvas()
        self.status_var.set(f"已生成白板 {w}x{h}")

    def import_image_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("图像", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")])
        if not path:
            return
        img = Image.open(path)
        # 选择导入模式（灰度化 / 二值化）
        mode = self.import_mode_var.get()
        entry_text = self.threshold_entry.get().strip()
        try:
            if mode == "灰度化":
                if img.mode != "L":
                    img = img.convert("L")
            else:
                # 二值化
                if img.mode != "L":
                    # color image -> LAB threshold
                    # parse entry: allow either default or 6 numbers
                    lab_vals = self._parse_lab_entry(entry_text)
                    if lab_vals is None:
                        lab_vals = DEFAULT_LAB
                    Lmax, Lmin, Amax, Amin, Bmax, Bmin = lab_vals
                    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2LAB)
                    lower = np.array([Lmin, Amin, Bmin], dtype=np.uint8)
                    upper = np.array([Lmax, Amax, Bmax], dtype=np.uint8)
                    mask = cv2.inRange(img_cv, lower, upper)  # 255 mask of selected region
                    img = Image.fromarray(mask).convert("L")
                else:
                    # grayscale -> parse min,max
                    gmin, gmax = self._parse_gray_entry(entry_text)
                    if gmin is None:
                        gmin, gmax = DEFAULT_GRAY_BIN
                    arr = np.array(img)
                    mask = np.where((arr >= gmin) & (arr <= gmax), 255, 0).astype(np.uint8)
                    img = Image.fromarray(mask, mode="L")
        except Exception as e:
            messagebox.showerror("处理错误", f"导入处理失败：{e}")
            return

        # 弹出裁剪/放大预览窗口（可手动裁剪或扩大）
        self._open_crop_preview(img)

    def _parse_lab_entry(self, text):
        # 支持 Lmax,Lmin,Amax,Amin,Bmax,Bmin 或 空
        if not text:
            return None
        parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip() != ""]
        if len(parts) != 6:
            return None
        try:
            vals = list(map(int, parts))
            return vals
        except:
            return None

    def _parse_gray_entry(self, text):
        if not text:
            return (None, None)
        parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip() != ""]
        if len(parts) != 2:
            return (None, None)
        try:
            return (int(parts[0]), int(parts[1]))
        except:
            return (None, None)

    # ----------------------------
    # 裁剪/放大预览窗口
    # ----------------------------
    def _open_crop_preview(self, img_pil):
        """
        打开一个独立窗口用于裁剪或放大（居中）图像。
        - 如果 img 大于目标分辨率，允许拖动框选裁剪区域；
        - 如果 img 小于目标分辨率，提供放大（fit to target）或居中按钮。
        """
        target_w, target_h = self.target_resolution
        preview = tk.Toplevel(self.root)
        preview.title("导入预览与裁剪")
        preview.geometry("900x700")

        frame = ttk.Frame(preview, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, bg="#222222")
        canvas.pack(fill=tk.BOTH, expand=True)

        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=6)
        ttk.Label(btn_frame, text=f"目标分辨率：{target_w} x {target_h}").pack(side=tk.LEFT, padx=6)

        # draw source img scaled to fit canvas but keep coords for crop mapping
        img = img_pil.copy()
        src_w, src_h = img.size

        # helper to draw image fitted to canvas
        def draw_fitted():
            canvas.delete("all")
            cw = canvas.winfo_width() or 800
            ch = canvas.winfo_height() or 600
            # compute scale to fit inside canvas
            ratio = min(cw / src_w, ch / src_h, 1.0)
            self._preview_scale = ratio
            self._preview_img_disp = img.resize((int(src_w * ratio), int(src_h * ratio)), Image.LANCZOS)
            self._preview_tk = ImageTk.PhotoImage(self._preview_img_disp)
            self._preview_pos = ((cw - self._preview_img_disp.width) // 2, (ch - self._preview_img_disp.height) // 2)
            canvas.create_image(self._preview_pos[0], self._preview_pos[1], anchor="nw", image=self._preview_tk)
            # draw target box scaled mapping
            if src_w > target_w or src_h > target_h:
                # show instruction
                canvas.create_text(10, 10, anchor="nw", text="拖动鼠标框选裁剪区域（放开确认）", fill="white")
            else:
                canvas.create_text(10, 10, anchor="nw", text="图像小于目标分辨率，可选择放大或居中", fill="white")
            # draw overlay showing target rectangle if same size as target (for reference)
            # Not drawing fixed box; cropping by user's rect
        # bind resize
        canvas.bind("<Configure>", lambda e: draw_fitted())
        draw_fitted()

        crop_rect = None
        start = None

        def on_down(e):
            nonlocal start, crop_rect
            start = (e.x, e.y)
            if crop_rect:
                canvas.delete(crop_rect)
                crop_rect = None

        def on_drag(e):
            nonlocal crop_rect
            if start:
                x1, y1 = start
                x2, y2 = e.x, e.y
                if crop_rect:
                    canvas.delete(crop_rect)
                crop_rect = canvas.create_rectangle(x1, y1, x2, y2, outline="red")

        def on_up(e):
            nonlocal crop_rect, img
            if not start:
                return
            x1, y1 = start
            x2, y2 = e.x, e.y
            # map to source image coords
            px, py = self._preview_pos
            sx1 = int(max(0, (min(x1, x2) - px) / self._preview_scale))
            sy1 = int(max(0, (min(y1, y2) - py) / self._preview_scale))
            sx2 = int(min(src_w, (max(x1, x2) - px) / self._preview_scale))
            sy2 = int(min(src_h, (max(y1, y2) - py) / self._preview_scale))
            if sx2 <= sx1 or sy2 <= sy1:
                return
            # crop to requested target size if crop area big enough, else pad
            cropped = img.crop((sx1, sy1, sx2, sy2))
            # if cropped bigger than target, center-crop to exact size
            if cropped.width >= target_w and cropped.height >= target_h:
                cropped = cropped.resize((target_w, target_h), Image.LANCZOS)
            else:
                # pad to target size with white
                new = Image.new("L", (target_w, target_h), 255)
                new.paste(cropped, ((target_w - cropped.width) // 2, (target_h - cropped.height) // 2))
                cropped = new
            self.image = cropped
            self.original_image = cropped.copy()
            self.push_history()
            preview.destroy()
            self.reset_view()
            self.redraw_canvas()

        canvas.bind("<ButtonPress-1>", on_down)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_up)

        # Buttons for small images
        def center_and_use():
            # pad to target
            new = Image.new("L", (target_w, target_h), 255)
            ox = (target_w - src_w) // 2
            oy = (target_h - src_h) // 2
            new.paste(img, (max(0, ox), max(0, oy)))
            self.image = new
            self.original_image = new.copy()
            self.push_history()
            preview.destroy()
            self.reset_view()
            self.redraw_canvas()

        def scale_up_to_target():
            new = img.resize((target_w, target_h), Image.LANCZOS)
            new = new.convert("L")
            self.image = new
            self.original_image = new.copy()
            self.push_history()
            preview.destroy()
            self.reset_view()
            self.redraw_canvas()

        ttk.Button(btn_frame, text="居中并补白（小图）", command=center_and_use).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="放大到目标（小图）", command=scale_up_to_target).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="取消", command=preview.destroy).pack(side=tk.RIGHT, padx=6)

    # ----------------------------
    # 视图相关函数：重置视图、缩放、平移、重绘
    # ----------------------------
    def reset_view(self):
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None

    def redraw_canvas(self, *_):
        # clear
        self.canvas.delete("all")
        if self.image is None:
            self.show_placeholder()
            return
        # compute displayed image with scale
        img_w, img_h = self.image.size
        disp_w = int(img_w * self.scale)
        disp_h = int(img_h * self.scale)
        disp = self.image.resize((disp_w, disp_h), Image.NEAREST if self.grid_var.get() else Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp)
        # compute center position in canvas_bg area
        cw = self.canvas_bg.winfo_width()
        ch = self.canvas_bg.winfo_height()
        # center offsets for canvas placement
        cx = max(10, (cw - disp_w) // 2 + self.offset_x)
        cy = max(10, (ch - disp_h) // 2 + self.offset_y)

        # place canvas size to exactly fit disp to allow proper mouse coords
        self.canvas.config(width=cw, height=ch)
        self.canvas.create_image(cx, cy, anchor="nw", image=self.tk_img, tags="img")
        # draw grey border for the image (visual only)
        self.canvas.create_rectangle(cx - 1, cy - 1, cx + disp_w + 1, cy + disp_h + 1, outline="gray", width=1)
        self.img_render_origin = (cx, cy)
        # draw pixel grid if requested
        if self.grid_var.get():
            self._draw_pixel_grid(cx, cy, img_w, img_h)
        # If user is dragging selection (preview snapshot exists), show selection overlay
        if self.preview_snapshot:
            # preview_snapshot is a PIL image to show while dragging; just draw it at origin
            pass
        self.status_var.set(f"图像: {img_w}x{img_h} 显示: {disp_w}x{disp_h} 缩放: {self.scale:.2f}")

    def _draw_pixel_grid(self, cx, cy, img_w, img_h):
        # draw rectangle per pixel scaled appropriately
        s = self.scale
        # to avoid huge number of rectangles at big images, only draw grid when scale large enough
        if s < 4:
            # draw coarse grid lines for pixel blocks
            # but user asked to "按下后把每一个像素区域框选出来", we'll draw only when scale >= 4, else skip with info
            self.canvas.create_text(cx + 8, cy + 12, anchor="nw", text="缩放到更大以显示像素网格", fill="red")
            return
        for i in range(img_w):
            x = cx + int(i * s)
            # vertical
            self.canvas.create_line(x, cy, x, cy + int(img_h * s), fill="#888", width=1)
        for j in range(img_h):
            y = cy + int(j * s)
            self.canvas.create_line(cx, y, cx + int(img_w * s), y, fill="#888", width=1)

    # ----------------------------
    # 编辑事件：鼠标绘制、平移、缩放
    # ----------------------------
    def set_tool(self, t):
        self.tool = t
        self.status_var.set(f"工具：{'画黑' if t == 'paint' else '擦除'}")

    def on_left_down(self, event):
        if self.image is None:
            return
        # start rectangle selection in image coords
        self.drag_start = (event.x, event.y)
        # snapshot for preview
        self.preview_snapshot = self.image.copy()

    def on_left_drag(self, event):
        if self.image is None or self.drag_start is None:
            return
        # show preview: create a temporary image with rectangle applied and re-render
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        ix0, iy0, ix1, iy1 = self._screen_to_image_rect(x0, y0, x1, y1)
        # clamp
        ix0 = max(0, min(self.image.width, ix0))
        ix1 = max(0, min(self.image.width, ix1))
        iy0 = max(0, min(self.image.height, iy0))
        iy1 = max(0, min(self.image.height, iy1))
        tmp_arr = np.array(self.preview_snapshot)
        if ix1 > ix0 and iy1 > iy0:
            if self.tool == "paint":
                tmp_arr[iy0:iy1, ix0:ix1] = 0
            else:
                tmp_arr[iy0:iy1, ix0:ix1] = 255
        self.tk_tmp = ImageTk.PhotoImage(Image.fromarray(tmp_arr))
        # draw this image at origin
        self.canvas.delete("tmpimg")
        ox, oy = self.img_render_origin
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_tmp, tags="tmpimg")

    def on_left_up(self, event):
        if self.image is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        ix0, iy0, ix1, iy1 = self._screen_to_image_rect(x0, y0, x1, y1)
        ix0 = max(0, min(self.image.width, ix0))
        ix1 = max(0, min(self.image.width, ix1))
        iy0 = max(0, min(self.image.height, iy0))
        iy1 = max(0, min(self.image.height, iy1))
        if ix1 > ix0 and iy1 > iy0:
            arr = np.array(self.image)
            if self.tool == "paint":
                arr[iy0:iy1, ix0:ix1] = 0
            else:
                arr[iy0:iy1, ix0:ix1] = 255
            self.image = Image.fromarray(arr)
            self.push_history()
        self.drag_start = None
        self.preview_snapshot = None
        self.canvas.delete("tmpimg")
        self.redraw_canvas()

    def on_middle_down(self, event):
        # start pan
        self.pan_start = (event.x, event.y, self.offset_x, self.offset_y)

    def on_middle_drag(self, event):
        if not self.pan_start:
            return
        x0, y0, ox0, oy0 = self.pan_start
        dx = event.x - x0
        dy = event.y - y0
        self.offset_x = ox0 + dx
        self.offset_y = oy0 + dy
        self.redraw_canvas()

    def on_middle_up(self, event):
        self.pan_start = None

    def on_mousewheel(self, event):
        if self.image is None:
            return
        # detect direction & delta
        if hasattr(event, "delta"):
            delta = event.delta
        elif event.num == 4:
            delta = 120
        else:
            delta = -120
        scale_factor = 1.0 + (0.0015 * delta)
        old_scale = self.scale
        new_scale = max(0.2, min(32.0, old_scale * scale_factor))

        # zoom towards mouse position: adjust offsets so that the point under cursor stays under cursor
        mx, my = event.x, event.y
        ox, oy = self.offset_x, self.offset_y
        # compute image origin
        img_origin_x, img_origin_y = self.img_render_origin
        # position of mouse relative to image origin in image-display-pixels
        rel_x = (mx - img_origin_x) / old_scale
        rel_y = (my - img_origin_y) / old_scale
        # new image origin so that rel_x*new_scale + new_origin_x ~= mx
        new_origin_x = mx - rel_x * new_scale
        new_origin_y = my - rel_y * new_scale
        # offset is difference between centered pos and new origin
        # Instead of recomputing complex center, we update offset so that image origin moves accordingly
        # Solve for new offset: assume old origin centered: ox is offset used in redraw
        # Set new offsets to shift image by delta_origin
        # approximate:
        dx_origin = (new_origin_x - img_origin_x)
        dy_origin = (new_origin_y - img_origin_y)
        self.scale = new_scale
        self.offset_x += dx_origin
        self.offset_y += dy_origin
        self.redraw_canvas()

    # ----------------------------
    # 坐标映射帮助函数
    # ----------------------------
    def _screen_to_image_rect(self, sx0, sy0, sx1, sy1):
        # map two screen coords (canvas) to image coords
        ox, oy = self.img_render_origin
        s = self.scale
        ix0 = int((min(sx0, sx1) - ox) / s)
        iy0 = int((min(sy0, sy1) - oy) / s)
        ix1 = int((max(sx0, sx1) - ox) / s)
        iy1 = int((max(sy0, sy1) - oy) / s)
        return ix0, iy0, ix1, iy1

    # ----------------------------
    # 历史（撤销/重做）
    # ----------------------------
    def push_history(self):
        # keep shallow copy stack up to reasonable size
        if self.image is None:
            return
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) <= 1:
            return
        self.redo_stack.append(self.undo_stack.pop())
        self.image = self.undo_stack[-1].copy()
        self.redraw_canvas()
        self.status_var.set("已撤销")

    def redo(self):
        if not self.redo_stack:
            return
        img = self.redo_stack.pop()
        self.undo_stack.append(img.copy())
        self.image = img.copy()
        self.redraw_canvas()
        self.status_var.set("已重做")

    # ----------------------------
    # 重置 / 保存
    # ----------------------------
    def reset_to_original(self):
        if self.original_image is None:
            return
        self.image = self.original_image.copy()
        self.push_history()
        self.redraw_canvas()
        self.status_var.set("已重置为导入/生成时状态")

    def save_mask(self):
        if self.image is None:
            messagebox.showerror("错误", "没有图像可保存")
            return
        # ensure binary
        arr = np.array(self.image)
        arr = np.where(arr > 128, 255, 0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG 文件", "*.png"), ("所有文件", "*.*")])
        if path:
            img.save(path)
            self.status_var.set(f"已保存：{path}")

    # ----------------------------
    # 其他
    # ----------------------------
    def show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(400, 200, text="请生成白板或导入图片\n（导入后可选择灰度化/二值化 / 裁剪）", fill="gray", font=("Arial", 18))

# ----------------------------
# 程序入口
# ----------------------------
def main():
    root = tk.Tk()
    app = MaskEditorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
