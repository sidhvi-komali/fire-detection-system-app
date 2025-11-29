# gui.py
"""
Finished Fire Detection GUI (CustomTkinter + Pillow)
Uses the model saved as 'models/fire_detection_model.keras'
"""

import math
import threading
from tkinter import filedialog
from PIL import Image
from customtkinter import CTkImage
import predict_image as predict_image 
import os
import traceback

try:
    import customtkinter as ctk
except Exception:
    raise ImportError("CustomTkinter is required. Install with: pip install customtkinter")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

SAMPLE_IMAGE_PATH = "fire-detection-system-app/dataset/0.jpg" 

def lerp_color(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))

def rgb_hex(rgb):
    return "#%02x%02x%02x" % rgb

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

class FireDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔥 Fire Detection System")
        self.geometry("1100x700")
        self.minsize(900, 600)

        self._tk_img = None
        self._anim_after_id = None
        self._last_confidence = 0.0

        self._arc_segments = []
        self._arc_segment_count = 120
        self._current_gradient = []

        self._no_fire_colors = []
        self._fire_colors = []

        self._setup_ui()
        self._prepare_gradients()

    def restart_app(self):
        try:
            self._stop_all_animations()
        except:
            pass
        self.destroy()
        new_app = FireDetectionApp()
        new_app.mainloop()

    def _setup_ui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)

        self.title_lbl = ctk.CTkLabel(self, text="🔥 Fire Detection System", font=("Helvetica", 28, "bold"))
        self.title_lbl.grid(row=0, column=0, columnspan=2, pady=(16, 12))

        self.left_frame = ctk.CTkFrame(self, corner_radius=8)
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=12)
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ne", padx=12, pady=12)

        self.upload_btn = ctk.CTkButton(btn_row, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=(0, 8))

        self.sample_btn = ctk.CTkButton(btn_row, text="Load Sample", command=self._load_sample)
        self.sample_btn.grid(row=0, column=1, padx=(0, 8))

        self.reset_btn = ctk.CTkButton(btn_row, text="Reset", command=self.reset, fg_color="#4F4F4F")
        self.reset_btn.grid(row=0, column=2)

        self.restart_btn = ctk.CTkButton(btn_row, text="Restart App", command=self.restart_app, fg_color="#444444")
        self.restart_btn.grid(row=0, column=3, padx=(8, 0))

        self.image_container = ctk.CTkFrame(self.left_frame, corner_radius=6)
        self.image_container.grid(row=0, column=0, sticky="nsew", padx=12, pady=(60, 12))
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_container, text="")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ctk.CTkFrame(self, corner_radius=16, fg_color="#2E2E2E", border_width=4, border_color="#000000")
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=12)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.result_lbl = ctk.CTkLabel(self.right_frame, text="", font=("Helvetica", 36, "bold"))
        self.result_lbl.grid(row=0, column=0, pady=(28, 8))

        self.sub_lbl = ctk.CTkLabel(self.right_frame, text="", font=("Helvetica", 12))
        self.sub_lbl.grid(row=1, column=0)

        self.conf_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.conf_row.grid(row=2, column=0, pady=(2, 0))

        self.conf_label = ctk.CTkLabel(self.conf_row, text="Confidence:", font=("Helvetica", 14, "bold"))
        self.conf_label.grid(row=0, column=0, padx=6)

        self.confidence_pct = ctk.CTkLabel(self.conf_row, text="0%", font=("Helvetica", 14, "bold"))
        self.confidence_pct.grid(row=0, column=1)

        self.canvas_size = 260
        self.center = self.canvas_size // 2
        self.ring_inner = 56
        self.ring_outer = 106

        self.circle_canvas = ctk.CTkCanvas(
            self.right_frame, width=self.canvas_size, height=self.canvas_size,
            bg="#2E2E2E", highlightthickness=0
        )
        self.circle_canvas.grid(row=3, column=0, pady=(14, 6))

        self.circle_bg = self.circle_canvas.create_oval(
            self.center - self.ring_outer, self.center - self.ring_outer,
            self.center + self.ring_outer, self.center + self.ring_outer,
            width=2, outline="#3a3a3a"
        )

        seg_angle = 360 / self._arc_segment_count
        for i in range(self._arc_segment_count):
            seg = self.circle_canvas.create_arc(
                self.center - self.ring_outer, self.center - self.ring_outer,
                self.center + self.ring_outer, self.center + self.ring_outer,
                start=90 - (i + 1) * seg_angle,
                extent=0,
                style="arc",
                width=self.ring_outer - self.ring_inner,
                outline="#000000"
            )
            self._arc_segments.append(seg)

        self.inner_circle = self.circle_canvas.create_oval(
            self.center - self.ring_inner, self.center - self.ring_inner,
            self.center + self.ring_inner, self.center + self.ring_inner,
            fill="#2E2E2E", outline="#2E2E2E"
        )

        self._percent_text = self.circle_canvas.create_text(
            self.center, self.center, text="0%", font=("Helvetica", 34, "bold"), fill="white"
        )

        self.loading_lbl = ctk.CTkLabel(self.right_frame, text="", font=("Helvetica", 12, "italic"))
        self.loading_lbl.grid(row=4, column=0)

        self.warning_lbl = ctk.CTkLabel(self, text="AI can make mistakes.", font=("Helvetica", 10), text_color="grey")
        self.warning_lbl.grid(row=2, column=1, sticky="e", padx=(0, 20), pady=(0, 12))

        self.left_frame.bind("<Configure>", self._on_left_resize)

    def _prepare_gradients(self):
        self._no_fire_colors = [lerp_color((50, 130, 200), (160, 230, 255), i / max(1, self._arc_segment_count - 1)) for i in range(self._arc_segment_count)]
        self._fire_colors = [lerp_color((160, 30, 30), (255, 120, 50), i / max(1, self._arc_segment_count - 1)) for i in range(self._arc_segment_count)]
        self._current_gradient = self._no_fire_colors.copy()

    def _display_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            container_w = max(200, self.image_container.winfo_width() - 24)
            container_h = max(200, self.image_container.winfo_height() - 24)
            img_ratio = img.width / img.height
            box_ratio = container_w / container_h
            if img_ratio > box_ratio:
                new_w = container_w
                new_h = int(new_w / img_ratio)
            else:
                new_h = container_h
                new_w = int(new_h * img_ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._tk_img = CTkImage(img, size=(new_w, new_h))
            self.image_label.configure(image=self._tk_img)
        except Exception as e:
            print("Error displaying image:", e)

    def _on_left_resize(self, event):
        if hasattr(self, "_last_image_path"):
            self._display_image(self._last_image_path)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self._last_image_path = path
        self._display_image(path)
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _load_sample(self):
        path = SAMPLE_IMAGE_PATH
        if not os.path.exists(path):
            self.loading_lbl.configure(text=f"Sample image not found at: {path}")
            return
        self._last_image_path = path
        self._display_image(path)
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def reset(self):
        self._stop_all_animations()
        self.image_label.configure(image=None)
        self.result_lbl.configure(text="")
        self.sub_lbl.configure(text="")
        for seg in self._arc_segments:
            self.circle_canvas.itemconfig(seg, extent=0, outline="#000000")
        self._set_percentage(0.0)
        try:
            self.right_frame.configure(border_color="#000000")
        except:
            pass
        if hasattr(self, "_last_image_path"):
            del self._last_image_path
        self.loading_lbl.configure(text="")

    def _stop_all_animations(self):
        if self._anim_after_id:
            try:
                self.after_cancel(self._anim_after_id)
            except:
                pass
            self._anim_after_id = None

    def _run_prediction(self, path):
        try:
            self.loading_lbl.configure(text="⏳ Analyzing image...")
            prediction = predict_image.predict(path)
            raw_prob = 0.0
            if "probability:" in prediction:
                try:
                    num = prediction.split("probability:")[1]
                    num = "".join([c for c in num if (c.isdigit() or c == "." or c == "-")])
                    raw_prob = float(num) if num else 0.0
                except Exception:
                    raw_prob = 0.0
            prob = max(0.0, min(1.0, float(raw_prob)))
            label = prediction.split(" (")[0].strip()
            self.after(0, lambda: self._update_result(label, prob))
        except Exception as error_obj:
            error_msg = str(error_obj)
            traceback_info = traceback.format_exc()
            self.after(0, lambda msg=error_msg, tb=traceback_info: self._show_error(msg, tb))

    def _update_result(self, label, prob):
        self._stop_all_animations()
        self.loading_lbl.configure(text="")
        display = label.upper()
        self.result_lbl.configure(text=display, text_color="white")
        is_fire = ("FIRE" in display) and ("NO" not in display)
        is_no_fire = ("NO FIRE" in display)
        if is_no_fire:
            self.sub_lbl.configure(text="No fire detected", text_color="#00BFFF")
            self._current_gradient = self._no_fire_colors.copy()
            self._animate_glow("#00BFFF")
        elif is_fire:
            self.sub_lbl.configure(text="Potential fire detected — stay alert!", text_color="#FF4D4D")
            self._current_gradient = self._fire_colors.copy()
            self._animate_glow("#FF4D4D")
        else:
            self.sub_lbl.configure(text="Analysis complete", text_color="grey")
            self._current_gradient = self._no_fire_colors.copy()
            self._animate_glow("#555555")
        self._animate_confidence(target=prob, duration_ms=1200)

    def _set_percentage(self, fraction):
        pct = int(max(0, min(1, fraction)) * 100)
        self.confidence_pct.configure(text=f"{pct}%")
        self.circle_canvas.itemconfig(self._percent_text, text=f"{pct}%")

    def _animate_confidence(self, target, duration_ms=1000):
        start = self._last_confidence
        end = max(0, min(1, target))
        self._last_confidence = end
        steps = max(60, int(duration_ms / 10))
        seg_angle = 360 / self._arc_segment_count
        def step(frame):
            t = frame / steps
            eased = math.sin((t * math.pi) / 2)
            val = start + (end - start) * eased
            self._set_percentage(val)
            degrees_to_show = val * 360
            for i, seg_id in enumerate(self._arc_segments):
                seg_start = 90 - i * seg_angle
                coverage = max(0, min(1, (degrees_to_show - i * seg_angle) / seg_angle))
                extent = -coverage * seg_angle - 0.5  
                color_rgb = self._current_gradient[i]
                try:
                    self.circle_canvas.itemconfig(seg_id, start=seg_start, extent=extent, outline=rgb_hex(color_rgb))
                except Exception:
                    self.circle_canvas.itemconfig(seg_id, start=seg_start, extent=extent, outline="#000000")
            if frame < steps:
                self._anim_after_id = self.after(int(duration_ms / steps), lambda: step(frame + 1))
            else:
                self._anim_after_id = None
        step(0)

    def _animate_glow(self, target_color_hex, steps=125):
        try:
            start_color = self.right_frame.cget("border_color")
            start = hex_to_rgb(start_color)
        except Exception:
            start = (0, 0, 0)
        end = hex_to_rgb(target_color_hex)
        def step(i):
            t = i / steps
            eased = math.sin((t * math.pi) / 2)
            r = int(start[0] + (end[0] - start[0]) * eased)
            g = int(start[1] + (end[1] - start[1]) * eased)
            b = int(start[2] + (end[2] - start[2]) * eased)
            try:
                self.right_frame.configure(border_color=rgb_hex((r, g, b)))
            except Exception:
                pass
            if i < steps:
                self.after(10, lambda: step(i + 1))
        step(0)

    def _show_error(self, msg, tb):
        self._stop_all_animations()
        self.loading_lbl.configure(text="")
        self.result_lbl.configure(text="ERROR", text_color="red")
        self.sub_lbl.configure(text="Prediction Failed: Check console for details", text_color="red")
        for seg in self._arc_segments:
            self.circle_canvas.itemconfig(seg, extent=0)
        self._set_percentage(0)
        self._animate_glow("#FF0000")
        print("\n--- ERROR DURING PREDICTION ---")
        print(msg)
        print(tb)
        print("-------------------------------\n")

    def on_closing(self):
        self._stop_all_animations()
        self.destroy()

def main():
    app = FireDetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
