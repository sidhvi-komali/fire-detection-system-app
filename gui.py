# gui.py
import math
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import predict_image as predict_image

try:
    import customtkinter as ctk
    USE_CTK = True
except Exception:
    raise ImportError("CustomTkinter is required. Please install it with `pip install customtkinter`.")


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FireDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔥 Fire Detection System")
        self.geometry("700x500")
        self._setup_ui()

    def _setup_ui(self):
        self.grid_rowconfigure((0, 1), weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title
        self.title_lbl = ctk.CTkLabel(self, text="🔥 Fire Detection System", font=("Helvetica", 24, "bold"))
        self.title_lbl.grid(row=0, column=0, pady=(20, 20))

        # Upload button
        self.upload_btn = ctk.CTkButton(self, text="Upload Image", command=self.upload_image, width=200)
        self.upload_btn.grid(row=1, column=0, pady=(0, 20))

        # Combined frame for image + result + confidence
        self.combined_frame = ctk.CTkFrame(self, width=650, height=500)
        self.combined_frame.grid(row=2, column=0, pady=(0, 20), padx=20, sticky="nsew")
        self.combined_frame.grid_propagate(False)

        # Image
        self.image_label = ctk.CTkLabel(self.combined_frame, text="")
        self.image_label.place(relx=0.5, rely=0.4, anchor="center")

        # Result label
        self.result_lbl = ctk.CTkLabel(self.combined_frame, text="", font=("Helvetica", 22, "bold"), text_color="white")
        self.result_lbl.place(relx=0.5, rely=0.75, anchor="center")

        # Confidence percentage label
        self.confidence_pct = ctk.CTkLabel(
            self.combined_frame, text="0%", font=("Helvetica", 16, "bold"), text_color="#2E6FC9"
        )
        self.confidence_pct.place(relx=0.85, rely=0.75, anchor="center")

        # Confidence bar inside the combined frame
        self.confidence = ctk.CTkProgressBar(self.combined_frame, width=600, height=25, progress_color="#ffffff")
        self.confidence.place(relx=0.5, rely=0.9, anchor="center")
        self.confidence.set(0.0)

        # Loading label
        self.loading_lbl = ctk.CTkLabel(self, text="", font=("Helvetica", 14, "italic"))
        self.loading_lbl.grid(row=3, column=0, pady=(0, 20))

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self._display_image(path)
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _display_image(self, path):
        img = Image.open(path).convert("RGB")
        max_width = 620
        w_percent = max_width / float(img.width) if img.width > max_width else 1.0
        new_width = int(img.width * w_percent)
        new_height = int(img.height * w_percent)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk_img)

    def _run_prediction(self, path):
        try:
            self.loading_lbl.configure(text="⏳ Analyzing image...")
            self.result_lbl.configure(text="")
            self.confidence.set(0.0)
            prediction_text = predict_image.predict(path)
            raw_prob = float(prediction_text.split("probability: ")[1].replace(")", ""))
            prob = 1 - raw_prob
            label = prediction_text.split(" (")[0]
            self.after(0, lambda: self._update_result(label, prob))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: self._show_error(str(e), tb))


    def _update_result(self, label, prob):
        self.loading_lbl.configure(text="")
        # Keep result text always white
        self.result_lbl.configure(text=label, text_color="white")
        # Animate confidence bar
        self._animate_confidence(label, prob)

    def _animate_confidence(self, label_text, target_prob):
        current = self.confidence.get()
        steps = 200
        duration = 1500
        step_time = duration / steps

        # Determine color transition: fire = white→red, no fire = white→green
        if "FIRE" in label_text.upper():
            start_color = (255, 255, 255)   # white
            end_color = (255, 77, 77)       # red (#ff4d4d)
        else:
            start_color = (255, 255, 255)   # white
            end_color = (40, 167, 69)       # green (#28a745)

        def rgb_to_hex(rgb):
            return "#%02x%02x%02x" % rgb

        def interpolate_color(start, end, t):
            return tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))

        def step(i):
            if i > steps:
                return
            t = i / steps
            eased_t = math.sin((t * math.pi) / 2)
            value = current + (target_prob - current) * eased_t
            self.confidence.set(value)
            self.confidence_pct.configure(text=f"{int(value*100)}%")
            color = rgb_to_hex(interpolate_color(start_color, end_color, eased_t))
            self.confidence.configure(progress_color=color)
            self.after(int(step_time), lambda: step(i + 1))

        # Reset the bar to white before animating
        self.confidence.configure(progress_color="#ffffff")
        step(0)

    def _show_error(self, msg, tb):
        self.loading_lbl.configure(text="")
        self.result_lbl.configure(text="Error", text_color="red")
        self.confidence.set(0.0)
        print("Prediction error:", msg)
        print(tb)


def main():
    app = FireDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
