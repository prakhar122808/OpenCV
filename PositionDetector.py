import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------- Config -----------------
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"

# ----------------- Helpers -----------------
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----------------- MediaPipe Tasks Setup -----------------
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

face_landmarker = FaceLandmarker.create_from_options(face_options)
hand_landmarker = HandLandmarker.create_from_options(hand_options)

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

    face_result = face_landmarker.detect_for_video(mp_image, timestamp)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

    expression = "No face"
    hand_infos = []

    # ----------------- Face: Expression -----------------
    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]

        # Key landmarks
        left_mouth = face[61]
        right_mouth = face[291]
        top_lip = face[13]
        bottom_lip = face[14]

        left_eye_outer = face[33]
        right_eye_outer = face[263]

        left_eye_top = face[159]
        left_eye_bottom = face[145]
        right_eye_top = face[386]
        right_eye_bottom = face[374]

        left_brow = face[70]
        right_brow = face[300]

        # Convert to pixels
        lm = (int(left_mouth.x * w), int(left_mouth.y * h))
        rm = (int(right_mouth.x * w), int(right_mouth.y * h))
        tl = (int(top_lip.x * w), int(top_lip.y * h))
        bl = (int(bottom_lip.x * w), int(bottom_lip.y * h))
        leo = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
        reo = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))

        let = (int(left_eye_top.x * w), int(left_eye_top.y * h))
        leb = (int(left_eye_bottom.x * w), int(left_eye_bottom.y * h))
        retp = (int(right_eye_top.x * w), int(right_eye_top.y * h))
        reb = (int(right_eye_bottom.x * w), int(right_eye_bottom.y * h))

        lb = (int(left_brow.x * w), int(left_brow.y * h))
        rb = (int(right_brow.x * w), int(right_brow.y * h))

        face_scale = dist(leo, reo)

        if face_scale > 1:
            mouth_width = dist(lm, rm)
            mouth_height = dist(tl, bl)

            mouth_width_norm = mouth_width / face_scale
            mouth_height_norm = mouth_height / face_scale

            left_eye_open = dist(let, leb)
            right_eye_open = dist(retp, reb)
            eye_open_norm = (left_eye_open + right_eye_open) / (2 * face_scale)

            # Mouth center
            center_lip_y = (tl[1] + bl[1]) // 2

            # Corner raise (smile)
            corner_offset = 0.03 * face_scale
            left_corner_up = lm[1] < center_lip_y - corner_offset
            right_corner_up = rm[1] < center_lip_y - corner_offset

            # Eyebrow raise (surprise)
            left_eye_center = ((let[0] + leb[0]) // 2, (let[1] + leb[1]) // 2)
            right_eye_center = ((retp[0] + reb[0]) // 2, (retp[1] + reb[1]) // 2)

            brow_eye_dist = (dist(lb, left_eye_center) + dist(rb, right_eye_center)) / 2
            brow_eye_norm = brow_eye_dist / face_scale

            # Decide expression
            if eye_open_norm < 0.02:
                expression = "Blinking"
            elif mouth_height_norm > 0.25 and brow_eye_norm > 0.12:
                expression = "Surprised"
            elif left_corner_up and right_corner_up and mouth_width_norm > 0.7:
                expression = "Smiling"
            else:
                expression = "Neutral"

        # Draw some points
        cv2.circle(frame, lm, 3, (0, 255, 0), -1)
        cv2.circle(frame, rm, 3, (0, 255, 0), -1)
        cv2.circle(frame, tl, 3, (0, 0, 255), -1)
        cv2.circle(frame, bl, 3, (0, 0, 255), -1)

    # ----------------- Hands: Placement & Gestures -----------------
    if hand_result.hand_landmarks:
        for idx, hand in enumerate(hand_result.hand_landmarks):
            # Draw landmarks
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Hand label
            hand_label = hand_result.handedness[idx][0].category_name

            # Wrist
            wx, wy = int(hand[0].x * w), int(hand[0].y * h)

            # Screen region
            if wx < w / 3:
                x_region = "Left"
            elif wx > 2 * w / 3:
                x_region = "Right"
            else:
                x_region = "Center"

            if wy < h / 3:
                y_region = "Top"
            elif wy > 2 * h / 3:
                y_region = "Bottom"
            else:
                y_region = "Middle"

            # Finger states
            fingers = {
                "index": (8, 6),
                "middle": (12, 10),
                "ring": (16, 14),
                "pinky": (20, 18),
            }

            extended = {}
            for name, (tip, pip) in fingers.items():
                extended[name] = hand[tip].y < hand[pip].y

            # Pinch
            tx, ty = int(hand[4].x * w), int(hand[4].y * h)
            ix, iy = int(hand[8].x * w), int(hand[8].y * h)
            pinch_dist = math.hypot(tx - ix, ty - iy)

            # Hand scale
            middle_mcp = (int(hand[9].x * w), int(hand[9].y * h))
            hand_scale = dist((wx, wy), middle_mcp) + 1e-6

            # Gesture classification
            if pinch_dist / hand_scale < 0.4:
                gesture = "Pinch"
            elif extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
                gesture = "Peace"
            elif extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
                gesture = "Point"
            elif not extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
                if hand[4].y < hand[3].y:
                    gesture = "Thumbs Up"
                else:
                    gesture = "Fist"
            elif all(extended.values()):
                gesture = "Open Palm"
            else:
                gesture = "Unknown"

            info = f"{hand_label} | Pos: {x_region}-{y_region} | Gesture: {gesture}"
            hand_infos.append(info)

            cv2.circle(frame, (wx, wy), 6, (255, 0, 0), -1)

    # ----------------- UI -----------------
    cv2.putText(frame, f"Expression: {expression}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    y0 = 80
    for i, info in enumerate(hand_infos):
        cv2.putText(frame, f"Hand {i+1}: {info}", (20, y0 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if not hand_infos:
        cv2.putText(frame, "No hands detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Expression + Multi-Hand Gesture Detector (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()