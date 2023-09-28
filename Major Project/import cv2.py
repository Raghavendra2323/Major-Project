import cv2
import mediapipe as mp
import pyautogui 
import time  # Import time for optional delay

pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions (you may need to adjust these)
w_scr, h_scr = pyautogui.size()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb= cv2.flip(img_rgb,1)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:    
        for landmarks in results.multi_hand_landmarks:
            for lm in landmarks.landmark:
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
                # Map hand position to screen coordinates
                x_scr = int(w_scr/w*x)
                y_scr = int(h_scr/h*y)
                # Move the mouse cursor
                pyautogui.moveTo(x_scr, y_scr, duration=0.2)  # You can adjust the duration
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()