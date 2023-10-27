import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

keyboard = Controller()

# Define button positions and labels
button_positions = {
    "Q": (50, 50), "W": (150, 50), "E": (250, 50), "R": (350, 50), "T": (450, 50),
    "Y": (550, 50), "U": (650, 50), "I": (750, 50), "O": (850, 50), "P": (950, 50),
    "A": (100, 150), "S": (200, 150), "D": (300, 150), "F": (400, 150), "G": (500, 150),
    "H": (600, 150), "J": (700, 150), "K": (800, 150), "L": (900, 150), ";": (1000, 150),
    "Z": (150, 250), "X": (250, 250), "C": (350, 250), "V": (450, 250), "B": (550, 250),
    "N": (650, 250), "M": (750, 250), ",": (850, 250), ".": (950, 250), "/": (1050, 250)
}

# Initialize finalText to store the typed text
finalText = ""

def drawButton(img, text, pos, clicked=False):
    x, y = pos
    w, h = 100, 100
    key_color = (255, 255, 255)  # White color for the keys
    highlight_color = (0, 255, 0) if clicked else (255, 0, 255)
    cv2.rectangle(img, pos, (x + w, y + h), highlight_color, cv2.FILLED)
    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, key_color, 4)


def main():
    global finalText  # Declare finalText as a global variable
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img)

        for key, pos in button_positions.items():
            if lmList:
                x, y = pos
                w, h = 100, 100

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    drawButton(img, key, pos, clicked=True)
                    l, _, _ = detector.findDistance(8, 12, img, draw=False)
                    if l < 30:
                        keyboard.press(key)
                        drawButton(img, key, pos, clicked=False)
                        finalText += key

        cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
