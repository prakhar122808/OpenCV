import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    cv2.imshow("Face Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
