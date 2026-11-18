"""
Eye Mouse: control your cursor with your eyes/face via a webcam.

  * Look around           -> cursor moves (after a 9-point calibration)
  * Wink LEFT eye (hold)  -> left click
  * Wink RIGHT eye (hold) -> right click
  * Close BOTH eyes ~2s   -> pause / resume tracking
  * Keys in preview window: q = quit, c = recalibrate, p = pause/resume

Install:
    pip install opencv-python mediapipe==0.10.14 pyautogui numpy

Run:
    python eye_mouse.py |||  py -3.12-64 eye_mouse.py

Tips: sit in even lighting (face lit from the front), keep the camera at about
eye level, and try to keep your head fairly steady after calibrating.
Mac: allow Camera + Accessibility (and Screen Recording if asked) for your terminal.
"""

import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# ----------------------------- SETTINGS (tweak these) -----------------------------
CAMERA_INDEX = 1          # try 1 if the wrong camera opens
SMOOTHING = 0.25          # 0-1. Lower = smoother but laggier cursor
MIN_MOVE_PX = 4           # ignore cursor jitter smaller than this
CLOSED_FRACTION = 0.65    # eye counts as "closed" below this fraction of your open-eye EAR
OPEN_FRACTION = 0.85      # eye counts as "open" above this fraction
WINK_HOLD_SEC = 0.25      # how long a wink must be held to click (avoids natural blinks)
BOTH_CLOSED_PAUSE_SEC = 2.0
CALIB_SETTLE_SEC = 1.2    # time to let your eyes reach each calibration dot
CALIB_COLLECT_SEC = 1.0   # time spent recording at each dot
# ------------------------------------------------------------------------------------

pyautogui.FAILSAFE = False   # cursor will visit screen corners; use 'q' to quit
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# MediaPipe Face Mesh landmark indices.
# In an UN-mirrored webcam frame, "33" is the user's RIGHT eye, "263" is the user's LEFT eye.
R_EYE = dict(outer=33, inner=133, ear=(33, 160, 158, 133, 153, 144), iris=468)
L_EYE = dict(outer=263, inner=362, ear=(362, 385, 387, 263, 373, 380), iris=473)
NOSE = 1


def ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6)


def eye_offset(pts, eye):
    """Iris position relative to the eye's center, normalized by eye width."""
    center = (pts[eye["outer"]] + pts[eye["inner"]]) / 2
    width = np.linalg.norm(pts[eye["outer"]] - pts[eye["inner"]]) + 1e-6
    return (pts[eye["iris"]] - center) / width


def read_face(cap, face_mesh):
    """Returns (frame, data) where data is None if no face found."""
    ok, frame = cap.read()
    if not ok:
        return None, None
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return frame, None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm])
    gaze = (eye_offset(pts, R_EYE) + eye_offset(pts, L_EYE)) / 2
    nose = pts[NOSE] / np.array([w, h])
    data = dict(
        feat=np.array([gaze[0], gaze[1], nose[0], nose[1]]),
        ear_r=ear(pts, R_EYE["ear"]),
        ear_l=ear(pts, L_EYE["ear"]),
        pts=pts,
    )
    return frame, data


class GazeModel:
    """Maps [gaze_x, gaze_y, nose_x, nose_y] -> screen (x, y) with ridge regression."""

    def fit(self, X, Y, lam=1e-2):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        A = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        reg = lam * np.eye(A.shape[1])
        reg[0, 0] = 0
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ Y)

    def predict(self, feat):
        a = np.r_[1.0, (feat - self.mu) / self.sd]
        return a @ self.W


def calibrate(cap, face_mesh):
    """Show 9 dots full-screen; record features while the user stares at each."""
    win = "Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fr = [0.08, 0.5, 0.92]
    targets = [(int(x * SCREEN_W), int(y * SCREEN_H)) for y in fr for x in fr]
    X, Y, ears = [], [], []

    for (tx, ty) in targets:
        feats = []
        start = time.time()
        while True:
            t = time.time() - start
            if t > CALIB_SETTLE_SEC + CALIB_COLLECT_SEC:
                break
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
            collecting = t > CALIB_SETTLE_SEC
            color = (0, 255, 0) if collecting else (0, 0, 255)
            cv2.circle(canvas, (tx, ty), 22, color, -1)
            cv2.circle(canvas, (tx, ty), 4, (0, 0, 0), -1)
            cv2.putText(canvas, "Look at the dot and keep your head still. ESC to cancel.",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(win)
                return None, None
            _, data = read_face(cap, face_mesh)
            if data is not None and collecting:
                feats.append(data["feat"])
                ears.append((data["ear_r"] + data["ear_l"]) / 2)
        if len(feats) < 5:
            cv2.destroyWindow(win)
            print("Couldn't see your face during calibration. Check lighting/camera and retry.")
            return None, None
        X.append(np.median(feats, axis=0))
        Y.append([tx, ty])

    cv2.destroyWindow(win)
    model = GazeModel()
    model.fit(np.array(X), np.array(Y, dtype=float))
    open_ear = float(np.percentile(ears, 70))  # typical eyes-open EAR
    return model, open_ear


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,  # refine_landmarks gives iris points
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Starting calibration in the next window...")
    model, open_ear = None, None
    while model is None:
        model, open_ear = calibrate(cap, face_mesh)
        if model is None:
            ans = input("Calibration failed/cancelled. Retry? [Y/n] ").strip().lower()
            if ans == "n":
                return

    close_thr, open_thr = open_ear * CLOSED_FRACTION, open_ear * OPEN_FRACTION
    print(f"Calibrated. Open-eye EAR ~{open_ear:.2f}. Wink to click. 'q' quits.")

    cur = np.array([SCREEN_W / 2, SCREEN_H / 2])
    paused = False
    wink_side, wink_start, armed = None, 0.0, True
    both_start = None
    status = ""

    while True:
        frame, data = read_face(cap, face_mesh)
        if frame is None:
            break
        now = time.time()
        disp = cv2.flip(frame, 1)  # mirror for a natural preview

        if data is not None:
            er, el = data["ear_r"], data["ear_l"]   # er = user's RIGHT eye, el = user's LEFT eye
            r_closed, l_closed = er < close_thr, el < close_thr
            r_open, l_open = er > open_thr, el > open_thr
            any_squint = (not r_open) or (not l_open)

            # --- both eyes closed for a while: toggle pause ---
            if r_closed and l_closed:
                both_start = both_start or now
                if now - both_start > BOTH_CLOSED_PAUSE_SEC:
                    paused = not paused
                    both_start = None
                    armed = False
                    time.sleep(0.5)
            else:
                both_start = None

            # --- cursor movement (frozen while eyes are closing so winks don't drag it) ---
            if not paused and not any_squint:
                target = np.clip(model.predict(data["feat"]), [0, 0], [SCREEN_W - 1, SCREEN_H - 1])
                new = cur + SMOOTHING * (target - cur)
                if np.linalg.norm(new - cur) > MIN_MOVE_PX:
                    cur = new
                    pyautogui.moveTo(int(cur[0]), int(cur[1]), _pause=False)

            # --- wink detection ---
            side = None
            if l_closed and r_open:
                side = "left"
            elif r_closed and l_open:
                side = "right"

            if side and armed and not paused:
                if side != wink_side:
                    wink_side, wink_start = side, now
                elif now - wink_start >= WINK_HOLD_SEC:
                    pyautogui.click(button=side)
                    status = f"{side.upper()} CLICK"
                    armed = False
            elif not side:
                wink_side = None
                if r_open and l_open:
                    armed = True  # re-arm only once both eyes are fully open again

            cv2.putText(disp, f"EAR L:{el:.2f} R:{er:.2f} (open~{open_ear:.2f})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "No face detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(disp, "PAUSED" if paused else "ACTIVE", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255) if paused else (0, 255, 0), 2)
        cv2.putText(disp, status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Eye Mouse (q=quit c=recalibrate p=pause)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("c"):
            m, e = calibrate(cap, face_mesh)
            if m is not None:
                model, open_ear = m, e
                close_thr, open_thr = open_ear * CLOSED_FRACTION, open_ear * OPEN_FRACTION

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()