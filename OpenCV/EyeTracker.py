import cv2
import mediapipe as mp
import pyautogui
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------- Smoothing ----------
alpha = 0.2
smooth_x, smooth_y = screen_w // 2, screen_h // 2
smooth_dx, smooth_dy = 0.0, 0.0

# ---------- Calibration ----------
calib_labels = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
calib_values = {}
calib_index = 0
calibrated = False

# ---------- Eye indices ----------
LEFT_EYE = {"outer":33, "inner":133, "top":159, "bottom":145, "iris":468}
RIGHT_EYE = {"outer":263, "inner":362, "top":386, "bottom":374, "iris":473}

def eye_gaze(face_landmarks, w, h, idx):
    o = to_px(face_landmarks.landmark[idx["outer"]], w, h)
    i = to_px(face_landmarks.landmark[idx["inner"]], w, h)
    t = to_px(face_landmarks.landmark[idx["top"]], w, h)
    b = to_px(face_landmarks.landmark[idx["bottom"]], w, h)
    ic = to_px(face_landmarks.landmark[idx["iris"]], w, h)

    eye_center = ((o[0] + i[0]) // 2, (t[1] + b[1]) // 2)
    eye_w = math.hypot(o[0] - i[0], o[1] - i[1])
    eye_h = math.hypot(t[0] - b[0], t[1] - b[1])

    if eye_w < 1 or eye_h < 1:
        return None

    dx = (ic[0] - eye_center[0]) / eye_w
    dy = (ic[1] - eye_center[1]) / eye_h
    return dx, dy

print("Calibration order:", calib_labels)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    info = "No face"
    dx = dy = 0.0

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            g1 = eye_gaze(face_landmarks, w, h, LEFT_EYE)
            g2 = eye_gaze(face_landmarks, w, h, RIGHT_EYE)
            if g1 is None or g2 is None:
                continue

            dx = (g1[0] + g2[0]) / 2.0
            dy = (g1[1] + g2[1]) / 2.0

            # Smooth gaze
            smooth_dx = 0.3 * dx + 0.7 * smooth_dx
            smooth_dy = 0.3 * dy + 0.7 * smooth_dy

            if not calibrated:
                target = calib_labels[calib_index]
                info = f"Look at {target} and press SPACE"
            else:
                info = "Tracking"

            if calibrated:
                # Use safe min/max mapping
                dx_min = calib_values["LEFT"][0]
                dx_max = calib_values["RIGHT"][0]
                dy_min = calib_values["UP"][1]
                dy_max = calib_values["DOWN"][1]

                # Add safety margins
                margin_x = 0.1 * (dx_max - dx_min + 1e-6)
                margin_y = 0.1 * (dy_max - dy_min + 1e-6)

                dx_min -= margin_x
                dx_max += margin_x
                dy_min -= margin_y
                dy_max += margin_y

                nx = (smooth_dx - dx_min) / (dx_max - dx_min + 1e-6)
                ny = (smooth_dy - dy_min) / (dy_max - dy_min + 1e-6)

                nx = clamp(nx, 0.0, 1.0)
                ny = clamp(ny, 0.0, 1.0)

                target_x = int(nx * screen_w)
                target_y = int(ny * screen_h)

                # Smooth cursor
                smooth_x = int(alpha * target_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * target_y + (1 - alpha) * smooth_y)

                pyautogui.moveTo(smooth_x, smooth_y)

    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"dx:{smooth_dx:.2f} dy:{smooth_dy:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Eye Tracker (Stable)", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = record calibration point
    if key == 32 and not calibrated and result.multi_face_landmarks:
        label = calib_labels[calib_index]
        calib_values[label] = (smooth_dx, smooth_dy)
        print("Captured", label, calib_values[label])
        calib_index += 1
        if calib_index >= len(calib_labels):
            calibrated = True
            print("Calibration complete:", calib_values)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
