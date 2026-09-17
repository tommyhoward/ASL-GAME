import math
import os
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageTk
import tkinter as tk

from new_project import (
    BASE,
    MODEL_PATH,
    NET_PATH,
    HOLD_FRAMES,
    REPEAT_COOLDOWN,
    SMOOTH_WINDOW,
    CONF_OK,
    CONF_SURE,
    CONF_MIN,
    FONT,
    CONNECTIONS,
    normalize,
    classify,
    predict_letter,
    text,
    open_camera,
    NET,
)


class ASLDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Reader")
        self.root.geometry("1000x760")
        self.root.minsize(800, 620)

        self.word = ""
        self.debug = False
        self.recent = deque(maxlen=SMOOTH_WINDOW)
        self.stable = None
        self.count = 0
        self.last_letter = ""
        self.last_time = 0.0
        self.t0 = time.monotonic()

        self.frame_label = tk.Label(root)
        self.frame_label.pack(fill="both", expand=True, padx=12, pady=(12, 8))

        self.status_var = tk.StringVar(value="ASL Reader ready")
        self.word_var = tk.StringVar(value="Word: ")
        self.current_var = tk.StringVar(value="Current: -")

        top_bar = tk.Frame(root)
        top_bar.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(top_bar, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(top_bar, textvariable=self.current_var, font=("Segoe UI", 11)).pack(side="right", padx=(0, 12))
        tk.Label(top_bar, textvariable=self.word_var, font=("Segoe UI", 13, "bold")).pack(side="right")

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        tk.Button(controls, text="Clear word", command=self.clear_word, width=16).pack(side="left", padx=(0, 8))
        tk.Button(controls, text="Toggle debug", command=self.toggle_debug, width=16).pack(side="left", padx=(0, 8))
        tk.Button(controls, text="Quit", command=self.close_app, width=10, bg="#f2d7d5", fg="#4a1f1f").pack(side="right")

        self.cap = open_camera()
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open the webcam.")

        self.options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.after(30, self.update_frame)

    def clear_word(self):
        self.word = ""
        self.word_var.set("Word: ")
        self.status_var.set("Word cleared")

    def toggle_debug(self):
        self.debug = not self.debug
        self.status_var.set("Debug on" if self.debug else "Debug off")

    def close_app(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.landmarker.close()
        except Exception:
            pass
        self.root.destroy()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Camera disconnected")
            self.root.after(100, self.close_app)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((time.monotonic() - self.t0) * 1000)
        result = self.landmarker.detect_for_video(image, timestamp)

        letter = None
        box = None

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]

            pix = []
            for lm in lms:
                pix.append((int(lm.x * w), int(lm.y * h)))
            for a, b in CONNECTIONS:
                cv2.line(frame, pix[a], pix[b], (0, 200, 120), 2)
            for p in pix:
                cv2.circle(frame, p, 4, (255, 255, 255), -1)

            xs = []
            ys = []
            for p in pix:
                xs.append(p[0])
                ys.append(p[1])
            box = (min(xs), max(ys))

            pts = normalize(lms)
            angle = math.atan2(lms[9].y - lms[0].y, lms[9].x - lms[0].x) + math.pi / 2

            if pts is not None:
                rule = classify(pts, angle)
            else:
                rule = None

            if NET is not None:
                letter, conf, top3 = predict_letter(lms)
                if len(top3) > 1:
                    runner_up = top3[1][1]
                else:
                    runner_up = 0.0
                confident = conf >= CONF_OK
                clear_winner = conf >= CONF_MIN and conf - runner_up >= 0.15

                rule_in_top3 = False
                for top_letter, _ in top3:
                    if top_letter == rule:
                        rule_in_top3 = True

                if conf < CONF_SURE and rule and rule_in_top3:
                    letter = rule
                elif not confident and not clear_winner:
                    letter = None

                if self.debug:
                    parts = []
                    for top_letter, p in top3:
                        parts.append(f"{top_letter} {p:.0%}")
                    text(frame, "net: " + "  ".join(parts), (10, 25), 0.55, (0, 255, 255), 1)
                    if rule:
                        rule_label = rule
                    else:
                        rule_label = "-"
                    if letter:
                        final_label = letter
                    else:
                        final_label = "-"
                    text(frame, f"rule: {rule_label}  final: {final_label}", (10, 50), 0.55, (0, 255, 255), 1)
            else:
                letter = rule

        self.recent.append(letter)

        votes = Counter()
        for x in self.recent:
            if x:
                votes[x] += 1
        if votes:
            smoothed = votes.most_common(1)[0][0]
        else:
            smoothed = None

        if smoothed and smoothed == self.stable:
            self.count += 1
        else:
            self.stable = smoothed
            self.count = 0

        if self.stable and self.count == HOLD_FRAMES:
            waited = time.monotonic() - self.last_time
            if self.stable != self.last_letter or waited > REPEAT_COOLDOWN:
                self.word += self.stable
                self.last_letter = self.stable
                self.last_time = time.monotonic()
            self.count = 0

        if box and smoothed:
            x1, y2 = box
            ly = min(y2 + 70, h - 60)
            size = cv2.getTextSize(smoothed, FONT, 2.2, 5)[0]
            bar_width = size[0]
            progress = min(self.count / HOLD_FRAMES, 1)
            fill_width = int(bar_width * progress)

            text(frame, smoothed, (x1, ly), 2.2, (0, 255, 255), 5)
            cv2.rectangle(frame, (x1, ly + 12), (x1 + bar_width, ly + 20), (80, 80, 80), -1)
            cv2.rectangle(frame, (x1, ly + 12), (x1 + fill_width, ly + 20), (0, 255, 0), -1)
            if self.word:
                text(frame, self.word, (x1, min(ly + 60, h - 15)), 1.1, (255, 255, 255), 3)

        if smoothed:
            self.current_var.set(f"Current: {smoothed}")
        else:
            self.current_var.set("Current: -")

        self.word_var.set(f"Word: {self.word}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((900, 560))
        photo = ImageTk.PhotoImage(frame)
        self.frame_label.configure(image=photo)
        self.frame_label.image = photo

        self.root.after(30, self.update_frame)


def main():
    root = tk.Tk()
    root.configure(bg="#f3f3f3")
    app = ASLDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\nSomething went wrong while starting the app. Press Enter to exit...")
