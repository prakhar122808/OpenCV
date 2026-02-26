import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------- Config -----------------
MODEL_PATH = "face_landmarker.task"

# ----------------- Helpers -----------------
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----------------- MediaPipe Tasks Setup -----------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ----------------- Camera -----------------
cap = cv2.VideoCapture(0)

timestamp = 0

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp += 1
    result = landmarker.detect_for_video(mp_image, timestamp)

    expression = "No face"

    if result.face_landmarks:
        face = result.face_landmarks[0]

        # ---- Key landmarks (MediaPipe indices) ----
        # Mouth
        left_mouth = face[61]
        right_mouth = face[291]
        top_lip = face[13]
        bottom_lip = face[14]

        # Eyes for face scale
        left_eye_outer = face[33]
        right_eye_outer = face[263]

        # Convert to pixels
        lm = (int(left_mouth.x * w), int(left_mouth.y * h))
        rm = (int(right_mouth.x * w), int(right_mouth.y * h))
        tl = (int(top_lip.x * w), int(top_lip.y * h))
        bl = (int(bottom_lip.x * w), int(bottom_lip.y * h))
        leo = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
        reo = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))

        # Face scale (eye distance) for normalization
        face_scale = dist(leo, reo)

        if face_scale > 1:
            # Measurements
            mouth_width = dist(lm, rm)
            mouth_height = dist(tl, bl)

            mouth_width_norm = mouth_width / face_scale
            mouth_height_norm = mouth_height / face_scale

            # Mouth center
            center_lip_y = (tl[1] + bl[1]) // 2

            # Corner raise (smile characteristic)
            corner_offset = 0.03 * face_scale
            left_corner_up = lm[1] < center_lip_y - corner_offset
            right_corner_up = rm[1] < center_lip_y - corner_offset

            # ---- Decision logic ----
            # Thresholds (tune if needed)
            if left_corner_up and right_corner_up and mouth_width_norm > 0.7:
                expression = "Smiling"
            else:
                expression = "Neutral"

        # ---- Draw debug points ----
        cv2.circle(frame, lm, 4, (0, 255, 0), -1)
        cv2.circle(frame, rm, 4, (0, 255, 0), -1)
        cv2.circle(frame, tl, 4, (0, 0, 255), -1)
        cv2.circle(frame, bl, 4, (0, 0, 255), -1)
        cv2.circle(frame, leo, 4, (255, 0, 0), -1)
        cv2.circle(frame, reo, 4, (255, 0, 0), -1)

    # ---- UI ----
    cv2.putText(frame, f"Expression: {expression}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Smile Detector (MediaPipe Tasks)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()