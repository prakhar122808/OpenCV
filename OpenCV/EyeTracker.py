import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---- Calibration storage ----
# We collect dx, dy for: center, left, right, up, down
calib_order = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
calib_data = {}
calib_index = 0
calibrated = False

# ---- Smoothing ----
alpha = 0.2  # 0.1–0.3 works well; lower = smoother, higher = more responsive
smooth_x, smooth_y = screen_w // 2, screen_h // 2

# ---- Gaze thresholds (used before calibration) ----
TH = 0.15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    info_text = ""
    dx = dy = 0.0

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # ---- Left eye landmarks ----
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            left_iris = face_landmarks.landmark[468]  # iris center

            leo = to_px(left_eye_outer, w, h)
            lei = to_px(left_eye_inner, w, h)
            let = to_px(left_eye_top, w, h)
            leb = to_px(left_eye_bottom, w, h)
            lic = to_px(left_iris, w, h)

            # Eye center
            eye_center = ((leo[0] + lei[0]) // 2, (let[1] + leb[1]) // 2)

            # Eye size for normalization
            eye_width = math.hypot(leo[0] - lei[0], leo[1] - lei[1])
            eye_height = math.hypot(let[0] - leb[0], let[1] - leb[1])

            if eye_width < 1 or eye_height < 1:
                continue

            # Normalized gaze vector
            dx = (lic[0] - eye_center[0]) / eye_width
            dy = (lic[1] - eye_center[1]) / eye_height

            # ---- Draw debug ----
            cv2.circle(frame, leo, 3, (0, 255, 0), -1)
            cv2.circle(frame, lei, 3, (0, 255, 0), -1)
            cv2.circle(frame, let, 3, (0, 255, 0), -1)
            cv2.circle(frame, leb, 3, (0, 255, 0), -1)
            cv2.circle(frame, lic, 5, (0, 0, 255), -1)       # iris
            cv2.circle(frame, eye_center, 5, (255, 0, 0), -1)  # eye center
            cv2.line(frame, eye_center, lic, (255, 255, 0), 2)

            # ---- Calibration flow ----
            if not calibrated:
                target = calib_order[calib_index]
                info_text = f"Look at {target} and press SPACE"
            else:
                info_text = "Tracking gaze -> cursor"

            # ---- Map gaze to screen after calibration ----
            if calibrated:
                # We have reference points
                c = calib_data["CENTER"]
                l = calib_data["LEFT"]
                r = calib_data["RIGHT"]
                u = calib_data["UP"]
                d = calib_data["DOWN"]

                # Compute normalized position between left-right and up-down
                # Avoid division by zero
                if abs(r[0] - l[0]) > 1e-6 and abs(d[1] - u[1]) > 1e-6:
                    nx = (dx - l[0]) / (r[0] - l[0])  # 0..1
                    ny = (dy - u[1]) / (d[1] - u[1])  # 0..1

                    nx = clamp(nx, 0.0, 1.0)
                    ny = clamp(ny, 0.0, 1.0)

                    target_x = int(nx * screen_w)
                    target_y = int(ny * screen_h)

                    # Exponential smoothing
                    smooth_x = int(alpha * target_x + (1 - alpha) * smooth_x)
                    smooth_y = int(alpha * target_y + (1 - alpha) * smooth_y)

                    pyautogui.moveTo(smooth_x, smooth_y)

    # ---- UI text ----
    cv2.putText(frame, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"dx: {dx:.2f} dy: {dy:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Eye Tracker (Gaze -> Mouse)", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE to capture calibration point
    if key == 32 and not calibrated and result.multi_face_landmarks:
        label = calib_order[calib_index]
        calib_data[label] = (dx, dy)
        calib_index += 1
        if calib_index >= len(calib_order):
            calibrated = True

    # ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
