import math
import os
import time
import traceback
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "hand_landmarker.task")
NET_PATH = os.path.join(BASE, "asl_model.npz")

if os.path.exists(NET_PATH):
    NET = np.load(NET_PATH)
else:
    NET = None

HOLD_FRAMES = 22
REPEAT_COOLDOWN = 1.2
SMOOTH_WINDOW = 7
CONF_OK = 0.60
CONF_SURE = 0.85
CONF_MIN = 0.35
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def normalize(lms):
    points = []
    for lm in lms:
        points.append([lm.x, lm.y])
    pts = np.array(points, np.float32)

    pts = pts - pts[0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        return None
    pts = pts / scale

    rot = -math.pi / 2 - math.atan2(pts[9][1], pts[9][0])
    c = math.cos(rot)
    s = math.sin(rot)
    rotation = np.array([[c, s], [-s, c]], np.float32)
    pts = pts @ rotation.T

    if pts[5][0] > pts[17][0]:
        pts[:, 0] = pts[:, 0] * -1

    return pts


def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))


def classify(pts, angle):
    states = []
    for tip in (8, 12, 16, 20):
        d = float(np.linalg.norm(pts[tip]))
        if d > 1.55:
            states.append("ext")
        elif d < 1.15:
            states.append("curl")
        else:
            states.append("half")

    idx = states[0]
    mid = states[1]
    ring = states[2]
    pinky = states[3]

    thumb = pts[4]
    n_ext = states.count("ext")
    n_curl = states.count("curl")

    side = abs(math.cos(angle)) > 0.72
    down = math.sin(angle) > 0.6
    thumb_out = thumb[0] < pts[5][0] - 0.45
    thumb_across = thumb[0] > pts[9][0] - 0.1
    thumb_to_index = dist(pts, 4, 8)

    if thumb_to_index < 0.35 and mid == "ext" and ring == "ext" and pinky == "ext":
        return "F"

    if pinky == "ext":
        ext_besides_pinky = n_ext - 1
    else:
        ext_besides_pinky = n_ext
    if thumb_out and pinky != "curl" and ext_besides_pinky == 0:
        return "Y"

    no_fingers_extended = n_ext == 0
    all_main_fingers_bent = (idx != "ext" and mid != "ext"
                             and ring != "ext" and pinky != "ext")

    if no_fingers_extended or (all_main_fingers_bent and n_curl < 4):
        if n_curl < 4:
            thumb_to_middle = dist(pts, 4, 12)
            pinch = (thumb_to_index + thumb_to_middle) / 2
            if pinch < 0.40:
                return "O"
            if 0.40 <= thumb_to_index < 1.0 and not thumb_across:
                return "C"
        if no_fingers_extended:
            if idx == "half" and thumb_to_index > 0.5 and n_curl >= 3:
                return "X"
            fingertip_height = (pts[8][1] + pts[12][1] + pts[16][1]) / 3
            if thumb[1] > fingertip_height + 0.12:
                return "E"
            if thumb_out or thumb[0] < pts[6][0] - 0.05:
                return "A"
            best_letter = "T"
            best_dist = dist(pts, 4, 6)
            for letter, joint in (("N", 10), ("M", 14), ("S", 18)):
                d = dist(pts, 4, joint)
                if d < best_dist:
                    best_dist = d
                    best_letter = letter
            return best_letter

    if n_ext == 4:
        if thumb_across or not thumb_out:
            return "B"
        return None

    if idx == "ext" and mid == "ext" and ring != "ext" and pinky != "ext":
        gap = dist(pts, 8, 12)
        if pts[8][0] > pts[12][0] and gap < 0.35:
            return "R"
        if dist(pts, 4, 10) < 0.45 or dist(pts, 4, 11) < 0.4:
            if down:
                return "P"
            return "K"
        if gap > 0.45:
            return "V"
        if side:
            return "H"
        return "U"

    if idx == "ext" and mid == "ext" and ring == "ext" and pinky != "ext":
        return "W"

    if idx == "ext" and mid != "ext" and ring != "ext" and pinky != "ext":
        if thumb_out:
            return "L"
        if down:
            return "Q"
        if side:
            return "G"
        return "D"

    if pinky == "ext" and idx != "ext" and mid != "ext" and ring != "ext":
        return "I"

    return None


def predict_letter(lms):
    points = []
    for lm in lms:
        points.append([lm.x, lm.y, lm.z])
    base = np.array(points, np.float32)

    best_letter = None
    best_conf = 0.0
    best_top3 = []

    for mirrored in (False, True):
        pts = base.copy()
        if mirrored:
            pts[:, 0] = 1.0 - pts[:, 0]

        flat = pts.reshape(1, -1)
        hidden = np.maximum(flat @ NET["w1"] + NET["b1"], 0)
        logits = hidden @ NET["w2"] + NET["b2"]
        exps = np.exp(logits - logits.max())
        probs = (exps / exps.sum())[0]

        conf = float(probs.max())
        if conf > best_conf:
            order = probs.argsort()[::-1][:3]
            top3 = []
            for i in order:
                letter = chr(65 + int(i))
                top3.append((letter, float(probs[i])))
            best_letter = top3[0][0]
            best_conf = conf
            best_top3 = top3

    return best_letter, best_conf, best_top3


def text(img, s, org, scale, color, thick):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, org, FONT, scale, color, thick)


def open_camera():
    attempts = [(0, cv2.CAP_MSMF), (0, cv2.CAP_DSHOW),
                (1, cv2.CAP_ANY), (2, cv2.CAP_ANY)]
    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    raise RuntimeError("Could not open a webcam. Close other camera apps and "
                       "check Windows camera privacy settings.")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} must sit next to this script.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = open_camera()
    word = ""
    debug = False
    recent = deque(maxlen=SMOOTH_WINDOW)
    stable = None
    count = 0
    last_letter = ""
    last_time = 0.0
    t0 = time.monotonic()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((time.monotonic() - t0) * 1000)
        result = landmarker.detect_for_video(image, timestamp)

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
            angle = math.atan2(lms[9].y - lms[0].y,
                               lms[9].x - lms[0].x) + math.pi / 2

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

                if debug:
                    parts = []
                    for top_letter, p in top3:
                        parts.append(f"{top_letter} {p:.0%}")
                    text(frame, "net: " + "  ".join(parts),
                         (10, 25), 0.55, (0, 255, 255), 1)
                    if rule:
                        rule_label = rule
                    else:
                        rule_label = "-"
                    if letter:
                        final_label = letter
                    else:
                        final_label = "-"
                    text(frame, f"rule: {rule_label}  final: {final_label}",
                         (10, 50), 0.55, (0, 255, 255), 1)
            else:
                letter = rule

        recent.append(letter)

        votes = Counter()
        for x in recent:
            if x:
                votes[x] += 1
        if votes:
            smoothed = votes.most_common(1)[0][0]
        else:
            smoothed = None

        if smoothed and smoothed == stable:
            count += 1
        else:
            stable = smoothed
            count = 0

        if stable and count == HOLD_FRAMES:
            waited = time.monotonic() - last_time
            if stable != last_letter or waited > REPEAT_COOLDOWN:
                word += stable
                last_letter = stable
                last_time = time.monotonic()
            count = 0

        if box and smoothed:
            x1, y2 = box
            ly = min(y2 + 70, h - 60)
            size = cv2.getTextSize(smoothed, FONT, 2.2, 5)[0]
            bar_width = size[0]
            progress = min(count / HOLD_FRAMES, 1)
            fill_width = int(bar_width * progress)

            text(frame, smoothed, (x1, ly), 2.2, (0, 255, 255), 5)
            cv2.rectangle(frame, (x1, ly + 12), (x1 + bar_width, ly + 20),
                          (80, 80, 80), -1)
            cv2.rectangle(frame, (x1, ly + 12), (x1 + fill_width, ly + 20),
                          (0, 255, 0), -1)
            if word:
                text(frame, word, (x1, min(ly + 60, h - 15)), 1.1,
                     (255, 255, 255), 3)

        cv2.rectangle(frame, (0, h - 44), (w, h), (30, 30, 30), -1)
        text(frame, f"Word: {word}", (12, h - 13), 0.9, (255, 255, 255), 2)
        cv2.imshow("ASL Reader", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            word += " "
        elif key == 8:
            word = word[:-1]
        elif key == ord("c"):
            word = ""
        elif key == ord("d"):
            debug = not debug

        if cv2.getWindowProperty("ASL Reader", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\nSomething went wrong (see error above). Press Enter to close...")
