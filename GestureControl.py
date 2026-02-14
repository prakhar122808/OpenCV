import cv2
import pyautogui
import time
import math
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
hand_detector = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# Click control
PINCH_CLICK_THRESHOLD = 0.04
CLICK_COOLDOWN = 0.3
last_click_time = 0

# Drag control
dragging = False

def is_fist(hand_landmarks):
    """
    Returns True if all fingers are folded.
    """
    fingers = [
        (8, 6),   # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18)  # pinky
    ]

    folded = 0
    for tip_id, pip_id in fingers:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[pip_id].y
        if tip_y > pip_y:  # finger folded
            folded += 1

    return folded == 4

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb)

    mode_text = "MOVE"
    status_color = (0, 255, 0)  # Green for MOVE

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmarks
            ix, iy = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y   # index tip
            tx, ty = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y   # thumb tip

            # Move cursor
            screen_x = int(ix * screen_w)
            screen_y = int(iy * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Distance for click (thumb + index)
            dist_click = math.hypot(ix - tx, iy - ty)

            now = time.time()

            # Detect fist for HOLD / DRAG
            fist = is_fist(hand_landmarks)

            if fist and not dragging:
                pyautogui.mouseDown()
                dragging = True
            elif not fist and dragging:
                pyautogui.mouseUp()
                dragging = False

            # Update mode based on state
            if dragging:
                mode_text = "DRAG"
                status_color = (0, 0, 255)  # Red
            else:
                mode_text = "MOVE"
                status_color = (0, 255, 0)  # Green

            # Detect CLICK (only if not dragging)
            if not dragging and dist_click < PINCH_CLICK_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = now
                mode_text = "CLICK"
                status_color = (0, 255, 255)  # Yellow

    # Draw HUD
    cv2.rectangle(frame, (10, 10), (220, 80), (0, 0, 0), -1)  # background box
    cv2.putText(frame, f"MODE: {mode_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Status indicator circle
    cv2.circle(frame, (200, 45), 10, status_color, -1)

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
