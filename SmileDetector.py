import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    expression = "No face"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # Cheek and eye landmarks (MediaPipe indices)
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            left_eye_lower = face_landmarks.landmark[145]
            right_eye_lower = face_landmarks.landmark[374]

            # Convert to pixels
            lc = (int(left_cheek.x * w), int(left_cheek.y * h))
            rc = (int(right_cheek.x * w), int(right_cheek.y * h))
            le = (int(left_eye_lower.x * w), int(left_eye_lower.y * h))
            re = (int(right_eye_lower.x * w), int(right_eye_lower.y * h))

            # Draw points
            cv2.circle(frame, lc, 5, (255, 0, 0), -1)  # left cheek (blue)
            cv2.circle(frame, rc, 5, (255, 0, 0), -1)  # right cheek (blue)
            cv2.circle(frame, le, 5, (0, 255, 255), -1)  # left eye lower (yellow)
            cv2.circle(frame, re, 5, (0, 255, 255), -1)  # right eye lower (yellow)

            # Draw vertical measurement lines
            cv2.line(frame, lc, le, (255, 255, 0), 2)
            cv2.line(frame, rc, re, (255, 255, 0), 2)

            # Distances (vertical is sufficient and more stable)
            left_cheek_eye_dist = abs(lc[1] - le[1])
            right_cheek_eye_dist = abs(rc[1] - re[1])

            # Show numeric values for debugging
            cv2.putText(frame, f"L CE: {left_cheek_eye_dist}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"R CE: {right_cheek_eye_dist}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # MediaPipe mouth landmarks
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            # Convert to pixel coordinates
            lm = (int(left_mouth.x * w), int(left_mouth.y * h))
            rm = (int(right_mouth.x * w), int(right_mouth.y * h))
            tl = (int(top_lip.x * w), int(top_lip.y * h))
            bl = (int(bottom_lip.x * w), int(bottom_lip.y * h))

            # Lower-cheek landmarks
            left_lower_cheek = face_landmarks.landmark[205]
            right_lower_cheek = face_landmarks.landmark[425]

            # Convert to pixels
            llc = (int(left_lower_cheek.x * w), int(left_lower_cheek.y * h))
            rlc = (int(right_lower_cheek.x * w), int(right_lower_cheek.y * h))

            # Mouth center (reference)
            center_lip_x = (tl[0] + bl[0]) // 2
            center_lip_y = (tl[1] + bl[1]) // 2
            center_lip = (center_lip_x, center_lip_y)

            # Vertical distances: cheek -> mouth center
            left_cheek_mouth_dist = llc[1] - center_lip_y
            right_cheek_mouth_dist = rlc[1] - center_lip_y

            # Mouth-corner raise (keep your existing logic)
            left_corner_up = lm[1] < center_lip_y - 5
            right_corner_up = rm[1] < center_lip_y - 5

            # Cheek lift condition: cheeks move up toward mouth center (distance decreases)
            CHEEK_LIFT_THRESHOLD = 0.02 * h  # tune 0.015–0.035

            cheeks_up = (left_cheek_mouth_dist < CHEEK_LIFT_THRESHOLD and
                        right_cheek_mouth_dist < CHEEK_LIFT_THRESHOLD)

            # Final decision (also keep a minimum mouth width to avoid tiny-mouth false positives)
            if left_corner_up and right_corner_up and cheeks_up and mouth_width > 0.08 * w:
                expression = "Smiling"
            else:
                expression = "Neutral"

            # Cheek and eye landmarks
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            left_eye_lower = face_landmarks.landmark[145]
            right_eye_lower = face_landmarks.landmark[374]

            # Convert to pixels
            lc = (int(left_cheek.x * w), int(left_cheek.y * h))
            rc = (int(right_cheek.x * w), int(right_cheek.y * h))
            le = (int(left_eye_lower.x * w), int(left_eye_lower.y * h))
            re = (int(right_eye_lower.x * w), int(right_eye_lower.y * h))

            # Distances: cheek to eye (vertical is enough, but Euclidean is fine)
            left_cheek_eye_dist = abs(lc[1] - le[1])
            right_cheek_eye_dist = abs(rc[1] - re[1])

            # Also keep your mouth-corner logic:
            center_lip_y = (tl[1] + bl[1]) // 2
            left_corner_up = lm[1] < center_lip_y - 5
            right_corner_up = rm[1] < center_lip_y - 5

            # Heuristic thresholds (tune these)
            CHEEK_LIFT_THRESHOLD = 0.1 * h  # depends on face size

            cheeks_up = (left_cheek_eye_dist < CHEEK_LIFT_THRESHOLD and
                        right_cheek_eye_dist < CHEEK_LIFT_THRESHOLD)

            # Final decision
            if left_corner_up and right_corner_up and cheeks_up and mouth_width > 0.08 * w:
                expression = "Smiling"
            else:
                expression = "Neutral"

            # Measurements
            mouth_width = dist(lm, rm)
            mouth_height = dist(tl, bl)

            # Center of lips
            center_lip_x = (tl[0] + bl[0]) // 2
            center_lip_y = (tl[1] + bl[1]) // 2

            # Check if corners are raised (smile characteristic)
            left_corner_up = lm[1] < center_lip_y - 5
            right_corner_up = rm[1] < center_lip_y - 5

            # Minimum width to avoid tiny/closed mouth false positives
            min_width = 0.25 * (rm[0] - lm[0] + abs(rm[0] - lm[0])) / 2  # or simpler: 0.08 * w

            if left_corner_up and right_corner_up and mouth_width > 0.08 * w:
                expression = "Smiling"
            else:
                expression = "Neutral"

            # Draw mouth points
            cv2.circle(frame, lm, 3, (0, 255, 0), -1)
            cv2.circle(frame, rm, 3, (0, 255, 0), -1)
            cv2.circle(frame, tl, 3, (0, 0, 255), -1)
            cv2.circle(frame, bl, 3, (0, 0, 255), -1)

            # Draw center reference
            cv2.circle(frame, (center_lip_x, center_lip_y), 3, (255, 0, 0), -1)

    # Show expression
    cv2.putText(frame, f"Expression: {expression}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Expression Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
