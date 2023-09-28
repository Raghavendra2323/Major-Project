import cv2


def return_camera_indices():
    index = -2
    arr = []
    i = 10
    while i>0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

print(return_camera_indices())