import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time


def compute_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
blink_threshold = 0.2
blink_count = 0
right_blink_count = 0
last_blink_time = time.time()
right_blink_timestamps = []

# Face Mesh indices for eye tracking
LEFT_EYE = [33, 133, 159, 145, 153, 158]
RIGHT_EYE = [362, 263, 386, 374, 382, 387]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_pts = []
            right_eye_pts = []

            for idx in LEFT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                left_eye_pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                right_eye_pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_eye_pts = np.array(left_eye_pts, dtype=np.int32)
            right_eye_pts = np.array(right_eye_pts, dtype=np.int32)

            if len(left_eye_pts) == 6 and len(right_eye_pts) == 6:
                left_EAR = compute_EAR(left_eye_pts)
                right_EAR = compute_EAR(right_eye_pts)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                if avg_EAR < blink_threshold:
                    if time.time() - last_blink_time > 1:  # Prevent rapid blinks
                        blink_count += 1
                        if blink_count % 2 == 0:
                            pyautogui.hotkey('ctrl', 'w')  # Close tab
                        else:
                            pyautogui.hotkey('ctrl', 't')  # Open new tab
                        last_blink_time = time.time()
                        print("Blink detected!")

                if right_EAR < blink_threshold:
                    current_time = time.time()
                    right_blink_timestamps.append(current_time)
                    right_blink_timestamps = [t for t in right_blink_timestamps if
                                              current_time - t < 1.0]  # Keep recent blinks within 1 sec

                    if len(right_blink_timestamps) == 1:
                        pyautogui.click()
                        print("Single Right Eye Blink - Left Click")
                    elif len(right_blink_timestamps) == 2:
                        pyautogui.rightClick()
                        print("Double Right Eye Blink - Right Click")
                        right_blink_timestamps = []  # Reset after double blink

                # Calculate eye center and map to screen
                eye_center_x = (left_eye_pts[:, 0].mean() + right_eye_pts[:, 0].mean()) / 2
                eye_center_y = (left_eye_pts[:, 1].mean() + right_eye_pts[:, 1].mean()) / 2

                cursor_x = np.interp(eye_center_x, [0, w], [0, screen_w])
                cursor_y = np.interp(eye_center_y, [0, h], [0, screen_h])
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
