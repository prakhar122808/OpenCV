import cv2
import pyautogui
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------- Config -----------------
MODEL_PATH = "hand_landmarker.task"

screen_w, screen_h = pyautogui.size()

PINCH_CLICK_THRESHOLD = 0.05   # normalized
CLICK_COOLDOWN = 0.3
last_click_time = 0

dragging = False

# ----------------- Helper functions -----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_fist(hand_landmarks):
    # landmarks are normalized [0,1]
    # Check if fingertips are below PIP joints (folded)
    fingers = [
        (8, 6),   # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18)  # pinky
    ]

    folded = 0
    for tip, pip in fingers:
        if hand_landmarks[tip].y > hand_landmarks[pip].y:
            folded += 1

    return folded == 4

# ----------------- MediaPipe Tasks setup -----------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ----------------- Camera -----------------
cap = cv2.VideoCapture(0)

mode_text = "MOVE"
status_color = (0, 255, 0)

timestamp = 0

# ----------------- Main loop -----------------
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

    mode_text = "MOVE"
    status_color = (0, 255, 0)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]  # first hand

        # Draw landmarks
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Index finger tip & thumb tip
        ix, iy = hand[8].x, hand[8].y
        tx, ty = hand[4].x, hand[4].y

        # Move cursor
        screen_x = int(ix * screen_w)
        screen_y = int(iy * screen_h)
        pyautogui.moveTo(screen_x, screen_y)

        # Pinch distance (normalized)
        pinch_dist = math.hypot(ix - tx, iy - ty)

        now = time.time()

        # Fist detection for drag
        fist = is_fist(hand)

        if fist and not dragging:
            pyautogui.mouseDown()
            dragging = True
        elif not fist and dragging:
            pyautogui.mouseUp()
            dragging = False

        if dragging:
            mode_text = "DRAG"
            status_color = (0, 0, 255)
        else:
            mode_text = "MOVE"
            status_color = (0, 255, 0)

        # Click (pinch) only if not dragging
        if not dragging and pinch_dist < PINCH_CLICK_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = now
            mode_text = "CLICK"
            status_color = (0, 255, 255)

    # ----------------- HUD -----------------
    cv2.rectangle(frame, (10, 10), (240, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"MODE: {mode_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.circle(frame, (220, 45), 10, status_color, -1)

    cv2.imshow("Hand Mouse (MediaPipe Tasks)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()