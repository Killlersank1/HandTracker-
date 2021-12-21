import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm

detector = htm.handDetector()
img = cv.imread('images/hand1_a_bot_seg_1_cropped.jpeg')

img = detector.findHands(img)
points = detector.findPosition(img)

print(points)

cv.imshow('Detected', img)
cv.waitKey(0)


