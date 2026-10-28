"""
Driver Drowsiness Detection System — desktop launcher (FYP I).
Asks for camera consent, then runs live OpenCV + MediaPipe detection.
"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent


def launch_streamlit() -> None:
    app = ROOT / "streamlit_app" / "streamlit_app.py"
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        cwd=str(ROOT),
    )


def run_desktop_detection() -> None:
    from drowsiness_core import DEFAULT_ALARM_PATH, DrowsinessEngine, play_alarm_async

    import cv2
    import time

    if not messagebox.askyesno(
        "Camera access",
        "Allow the app to use your webcam for drowsiness detection?\n\n"
        "Choose Yes to open the camera and start monitoring.",
    ):
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera error", "Could not open webcam. Check connections and permissions.")
        return

    engine = DrowsinessEngine()
    last_alert = 0.0
    window = "Driver Drowsiness Detection — press Q to quit, R to reset"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = engine.process_bgr_frame(frame)
            if engine.metrics.status == "DROWSY":
                now = time.time()
                if now - last_alert >= 4.0:
                    last_alert = now
                    engine.metrics.alert_count += 1
                    play_alarm_async(DEFAULT_ALARM_PATH)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                engine = DrowsinessEngine()
                last_alert = 0.0
    finally:
        engine.close()
        cap.release()
        cv2.destroyAllWindows()


def run_dlib_blink() -> None:
    model = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
    if not model.is_file():
        messagebox.showerror(
            "Model missing",
            "Download shape_predictor_68_face_landmarks.dat from dlib.net,\n"
            "extract it, and place it in the models/ folder.\n\n"
            "Use 'Start live detection' for MediaPipe mode (no extra download).",
        )
        return
    script = ROOT / "blinkDetect.py"
    subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT))


def run_face_detection() -> None:
    if not messagebox.askyesno("Camera access", "Open webcam for Haar cascade face detection?"):
        return
    subprocess.Popen([sys.executable, str(ROOT / "face-try.py")], cwd=str(ROOT))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Driver Drowsiness Detection System")
        self.geometry("720x520")
        self.minsize(640, 480)
        self.configure(bg="#0f172a")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#0f172a")
        style.configure(
            "Title.TLabel",
            background="#0f172a",
            foreground="#f8fafc",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Sub.TLabel",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 12, "bold"),
            padding=(16, 12),
        )

        frame = ttk.Frame(self, style="Dark.TFrame", padding=28)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Driver Drowsiness Detection", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="KFUEIT · FYP I · Real-time EAR monitoring & alerts",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 24))

        card = tk.Frame(frame, bg="#1e293b", highlightbackground="#334155", highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 20))
        inner = tk.Frame(card, bg="#1e293b", padx=20, pady=20)
        inner.pack(fill=tk.X)

        tk.Label(
            inner,
            text="Recommended: start live detection after allowing camera access.",
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=620,
        ).pack(anchor="w")

        btn_row = tk.Frame(inner, bg="#1e293b")
        btn_row.pack(fill=tk.X, pady=(16, 0))

        ttk.Button(
            btn_row,
            text="Start live detection (webcam)",
            style="Action.TButton",
            command=run_desktop_detection,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            btn_row,
            text="Open web dashboard",
            style="Action.TButton",
            command=launch_streamlit,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            btn_row,
            text="Face detection (OpenCV)",
            style="Action.TButton",
            command=run_face_detection,
        ).pack(side=tk.LEFT)

        ttk.Button(
            frame,
            text="Classic blink script (dlib + EAR)",
            style="Action.TButton",
            command=run_dlib_blink,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            frame,
            text="Quit",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(24, 4))
        ttk.Button(frame, text="Exit application", command=self.destroy).pack(anchor="w")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
