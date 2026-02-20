import cv2
import mediapipe as mp
import math

# ----------------- Setup -----------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    expression = "No face"
    hand_infos = []  # store info for each hand

    # ----------------- Face: Expression -----------------
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            # Key mouth landmarks
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            lm = to_px(left_mouth, w, h)
            rm = to_px(right_mouth, w, h)
            tl = to_px(top_lip, w, h)
            bl = to_px(bottom_lip, w, h)

            mouth_width = dist(lm, rm)
            mouth_height = dist(tl, bl)

            # Mouth center Y
            center_lip_y = (tl[1] + bl[1]) // 2

            # Face scale (eye distance)
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            leo = to_px(left_eye_outer, w, h)
            reo = to_px(right_eye_outer, w, h)
            face_scale = dist(leo, reo)

            # Eye landmarks for blink
            left_eye_top = to_px(face_landmarks.landmark[159], w, h)
            left_eye_bottom = to_px(face_landmarks.landmark[145], w, h)
            right_eye_top = to_px(face_landmarks.landmark[386], w, h)
            right_eye_bottom = to_px(face_landmarks.landmark[374], w, h)

            left_eye_open = dist(left_eye_top, left_eye_bottom)
            right_eye_open = dist(right_eye_top, right_eye_bottom)

            if face_scale > 1:
                # Normalized measures
                mouth_width_norm = mouth_width / face_scale
                mouth_height_norm = mouth_height / face_scale
                eye_open_norm = (left_eye_open + right_eye_open) / (2 * face_scale)

                # Corner raise for smile
                corner_offset = 0.03 * face_scale
                left_corner_up = lm[1] < center_lip_y - corner_offset
                right_corner_up = rm[1] < center_lip_y - corner_offset

                # Eyebrow raise for surprise
                left_brow = to_px(face_landmarks.landmark[70], w, h)
                right_brow = to_px(face_landmarks.landmark[300], w, h)
                left_eye_center = ((left_eye_top[0] + left_eye_bottom[0]) // 2,
                                   (left_eye_top[1] + left_eye_bottom[1]) // 2)
                right_eye_center = ((right_eye_top[0] + right_eye_bottom[0]) // 2,
                                    (right_eye_top[1] + right_eye_bottom[1]) // 2)

                brow_eye_dist = (dist(left_brow, left_eye_center) + dist(right_brow, right_eye_center)) / 2
                brow_eye_norm = brow_eye_dist / face_scale

                # Blink detection
                if eye_open_norm < 0.02:
                    expression = "Blinking"
                # Surprise: mouth open + eyebrows up
                elif mouth_height_norm > 0.25 and brow_eye_norm > 0.12:
                    expression = "Surprised"
                # Smile
                elif left_corner_up and right_corner_up and mouth_width_norm > 0.7:
                    expression = "Smiling"
                else:
                    expression = "Neutral"

            # Draw some face points
            cv2.circle(frame, lm, 3, (0, 255, 0), -1)
            cv2.circle(frame, rm, 3, (0, 255, 0), -1)
            cv2.circle(frame, tl, 3, (0, 0, 255), -1)
            cv2.circle(frame, bl, 3, (0, 0, 255), -1)

    # ----------------- Hands: Placement & Gestures -----------------
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Hand label
            hand_label = hand_result.multi_handedness[idx].classification[0].label

            # Wrist position
            wrist = hand_landmarks.landmark[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

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

            # Finger states (tip vs pip)
            fingers = {
                "index": (8, 6),
                "middle": (12, 10),
                "ring": (16, 14),
                "pinky": (20, 18),
            }

            extended = {}
            for name, (tip, pip) in fingers.items():
                tip_y = hand_landmarks.landmark[tip].y
                pip_y = hand_landmarks.landmark[pip].y
                extended[name] = tip_y < pip_y  # True if finger up

            # Thumb-index distance for pinch
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            pinch_dist = math.hypot(tx - ix, ty - iy)

            # Estimate hand size for normalization
            middle_mcp = hand_landmarks.landmark[9]
            wrist_px = (wx, wy)
            middle_mcp_px = (int(middle_mcp.x * w), int(middle_mcp.y * h))
            hand_scale = dist(wrist_px, middle_mcp_px) + 1e-6

            # Gesture classification
            if pinch_dist / hand_scale < 0.4:
                gesture = "Pinch"
            elif extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
                gesture = "Peace"
            elif extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
                gesture = "Point"
            elif not extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
                # Check thumb for thumbs up
                thumb_tip_y = hand_landmarks.landmark[4].y
                thumb_ip_y = hand_landmarks.landmark[3].y
                if thumb_tip_y < thumb_ip_y:
                    gesture = "Thumbs Up"
                else:
                    gesture = "Fist"
            elif all(extended.values()):
                gesture = "Open Palm"
            else:
                gesture = "Unknown"

            info = f"{hand_label} | Pos: {x_region}-{y_region} | Gesture: {gesture}"
            hand_infos.append(info)

            # Draw wrist
            cv2.circle(frame, (wx, wy), 6, (255, 0, 0), -1)

    # ----------------- UI Text -----------------
    cv2.putText(frame, f"Expression: {expression}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    y0 = 80
    for i, info in enumerate(hand_infos):
        cv2.putText(frame, f"Hand {i+1}: {info}", (20, y0 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if not hand_infos:
        cv2.putText(frame, "No hands detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Expression + Multi-Hand Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
