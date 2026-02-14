import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Helper
def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    gaze = "No face"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # ---- Left eye landmarks (MediaPipe) ----
            # Eye corners
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]

            # Upper & lower eyelid mid points
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]

            # Iris center (requires refine_landmarks=True)
            left_iris = face_landmarks.landmark[468]

            # Convert to pixels
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

            # Gaze vector (normalized)
            dx = (lic[0] - eye_center[0]) / eye_width
            dy = (lic[1] - eye_center[1]) / eye_height

            # Thresholds (tune)
            TH = 0.15

            if dx < -TH:
                gaze = "LEFT"
            elif dx > TH:
                gaze = "RIGHT"
            elif dy < -TH:
                gaze = "UP"
            elif dy > TH:
                gaze = "DOWN"
            else:
                gaze = "CENTER"

            # ---- Debug drawing ----
            cv2.circle(frame, leo, 3, (0, 255, 0), -1)
            cv2.circle(frame, lei, 3, (0, 255, 0), -1)
            cv2.circle(frame, let, 3, (0, 255, 0), -1)
            cv2.circle(frame, leb, 3, (0, 255, 0), -1)

            cv2.circle(frame, lic, 5, (0, 0, 255), -1)       # iris center (red)
            cv2.circle(frame, eye_center, 5, (255, 0, 0), -1)  # eye center (blue)

            cv2.line(frame, eye_center, lic, (255, 255, 0), 2)

            cv2.putText(frame, f"dx: {dx:.2f} dy: {dy:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Gaze: {gaze}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Eye Tracker (Gaze Direction)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
