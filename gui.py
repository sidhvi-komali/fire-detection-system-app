# gui.py
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import traceback
from PIL import Image, ImageTk
import predict_image

# Try to use CustomTkinter for a modern UI; otherwise fall back to Tkinter
try:
    import customtkinter as ctk
    USE_CTK = True
except Exception:
    import tkinter as tk
    from tkinter import filedialog, ttk
    USE_CTK = False


if USE_CTK:
    ctk.set_appearance_mode("System")  # "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")


class FireDetectionAppCTK(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔥 Fire Detection System")
        self.geometry("700x820")
        self._setup_ui()

    def _setup_ui(self):
        self.grid_rowconfigure((0, 1), weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(self, text="🔥 Fire Detection System", font=("Helvetica", 24, "bold"))
        self.title_lbl.grid(row=0, column=0, pady=(18, 6))

        top_row = ctk.CTkFrame(self)
        top_row.grid(row=1, column=0, pady=(6, 10), padx=18, sticky="ew")
        top_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.upload_btn = ctk.CTkButton(top_row, text="Upload Image", command=self.upload_image, width=160)
        self.upload_btn.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        self.theme_btn = ctk.CTkSegmentedButton(top_row, values=["Light", "Dark"], command=self._on_theme_change)
        # set it to match system default
        self.theme_btn.set("Light" if ctk.get_appearance_mode() == "Light" else "Dark")
        self.theme_btn.grid(row=0, column=2, padx=8, pady=8, sticky="e")

        self.image_frame = ctk.CTkFrame(self, width=620, height=380)
        self.image_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.image_frame.grid_propagate(False)

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        bottom = ctk.CTkFrame(self)
        bottom.grid(row=3, column=0, padx=20, pady=(10, 18), sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        self.loading_lbl = ctk.CTkLabel(bottom, text="", font=("Helvetica", 13, "italic"))
        self.loading_lbl.grid(row=0, column=0, sticky="w", padx=(6, 0), pady=(6, 6))

        self.result_lbl = ctk.CTkLabel(bottom, text="", font=("Helvetica", 20, "bold"))
        self.result_lbl.grid(row=1, column=0, sticky="w", padx=6)

        self.confidence = ctk.CTkProgressBar(bottom, width=560)
        self.confidence.grid(row=2, column=0, padx=6, pady=(10, 6))
        self.confidence.set(0.0)

    def _on_theme_change(self, value):
        if value == "Dark":
            ctk.set_appearance_mode("Dark")
        else:
            ctk.set_appearance_mode("Light")

    def upload_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self._display_image(path)
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _display_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((620, 380), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk_img)

    def _run_prediction(self, path):
        try:
            self.loading_lbl.configure(text="⏳ Analyzing image...")
            self.result_lbl.configure(text="")
            self.confidence.set(0.0)
            prediction_text = predict_image.predict(path)
            # Expected format: "🔥 FIRE DETECTED (probability: 0.8754)"
            prob = float(prediction_text.split("probability: ")[1].replace(")", ""))
            label = prediction_text.split(" (")[0]
            self.after(0, lambda: self._update_result(label, prob))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: self._show_error(str(e), tb))

    def _update_result(self, label, prob):
        self.loading_lbl.configure(text="")
        color = "#ff4d4d" if "FIRE" in label else "#28a745"
        self.result_lbl.configure(text=label, text_color=color)
        self.confidence.set(max(0.0, min(1.0, prob)))

    def _show_error(self, msg, tb):
        self.loading_lbl.configure(text="")
        self.result_lbl.configure(text="Error", text_color="#ff3333")
        self.confidence.set(0.0)
        print("Prediction error:", msg)
        print(tb)


class FireDetectionAppTK(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔥 Fire Detection System")
        self.geometry("700x820")
        self._light_bg = "#F4F4F4"
        self._light_panel = "#E8E8E8"
        self._dark_bg = "#121212"
        self._dark_panel = "#1e1e1e"
        self._use_dark = False
        self.configure(bg=self._light_bg)
        self._setup_ui()

    def _setup_ui(self):
        from tkinter import ttk
        self.title_lbl = tk.Label(self, text="🔥 Fire Detection System", font=("Helvetica", 24, "bold"), bg=self._light_bg)
        self.title_lbl.pack(pady=(18, 8))

        top_frame = tk.Frame(self, bg=self._light_bg)
        top_frame.pack(fill="x", padx=18, pady=(4, 8))
        self.upload_btn = ttk.Button(top_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side="left", padx=(0, 8))
        self.theme_btn = ttk.Button(top_frame, text="Toggle Dark Mode", command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=(8, 0))

        self.image_frame = tk.Frame(self, bg=self._light_panel, width=620, height=380)
        self.image_frame.pack(padx=20, pady=10)
        self.image_frame.pack_propagate(False)
        self.image_label = tk.Label(self.image_frame, bg=self._light_panel)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        bottom = tk.Frame(self, bg=self._light_bg)
        bottom.pack(fill="x", padx=20, pady=(6, 18))

        self.loading_lbl = tk.Label(bottom, text="", font=("Helvetica", 13, "italic"), bg=self._light_bg, fg="gray")
        self.loading_lbl.pack(anchor="w", pady=(6, 2))

        self.result_lbl = tk.Label(bottom, text="", font=("Helvetica", 20, "bold"), bg=self._light_bg)
        self.result_lbl.pack(anchor="w", pady=(2, 8))

        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence = ttk.Progressbar(bottom, length=560, mode="determinate", variable=self.confidence_var)
        self.confidence.pack(pady=(6, 2))

    def toggle_theme(self):
        self._use_dark = not self._use_dark
        if self._use_dark:
            bg = self._dark_bg; panel = self._dark_panel; fg = "white"
            self.theme_btn.config(text="Switch to Light Mode")
        else:
            bg = self._light_bg; panel = self._light_panel; fg = "black"
            self.theme_btn.config(text="Switch to Dark Mode")
        self.configure(bg=bg)
        self.title_lbl.config(bg=bg, fg=fg)
        for widget in (self.image_frame, self.image_label):
            widget.config(bg=panel)
        self.loading_lbl.config(bg=bg, fg="lightgray" if self._use_dark else "gray")
        self.result_lbl.config(bg=bg, fg=fg)

    def upload_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self._display_image(path)
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _display_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((620, 380), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk_img)

    def _run_prediction(self, path):
        try:
            self.loading_lbl.config(text="⏳ Analyzing image...")
            self.result_lbl.config(text="")
            self.confidence_var.set(0.0)
            prediction_text = predict_image.predict(path)
            prob = float(prediction_text.split("probability: ")[1].replace(")", ""))
            label = prediction_text.split(" (")[0]
            self.after(0, lambda: self._update_result(label, prob))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: self._show_error(str(e), tb))

    def _update_result(self, label, prob):
        self.loading_lbl.config(text="")
        color = "blue" if "FIRE" in label else "Red"
        self.result_lbl.config(text=label, fg=color)
        self.confidence_var.set(max(0.0, min(1.0, prob)))

    def _show_error(self, msg, tb):
        self.loading_lbl.config(text="")
        self.result_lbl.config(text="Error", fg="red")
        self.confidence_var.set(0.0)
        print("Prediction error:", msg)
        print(tb)


def main():
    if USE_CTK:
        app = FireDetectionAppCTK()
        app.mainloop()
    else:
        app = FireDetectionAppTK()
        app.mainloop()


if __name__ == "__main__":
    main()
