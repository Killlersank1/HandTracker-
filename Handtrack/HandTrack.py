import cv2 as cv
import mediapipe as mp
import time

# For Video Capturing
cap = cv.VideoCapture(0)

# For Hand Tracking
mpHands = mp.solutions.hands
hand = mpHands.Hands()

# For Drawing Into the Hand
mpDraw = mp.solutions.drawing_utils

# For Calculating Frame Rate
cTime = 0
pTime = 0

# Loop to Show Video Frame by Frame
while True:
    # For Reading the Video
    isTrue, frame = cap.read()

    # Converting the BGR image to RGB
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Results Object for Hand (Takes RGM image)
    results = hand.process(imgRGB)

    # # To track the hand coordinates
    #     print(results.multi_hand_landmarks)

    # Loop to Mark and Draw Landmarks in Hands
    if results.multi_hand_landmarks:
        for EachHand in results.multi_hand_landmarks:
            points = []
            for id, lm in enumerate(EachHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # #Making Circle on the tip of thumb
                # if id == 4:
                #     cv.circle(frame, (cx,cy), 25, (255,255,0), cv.FILLED)

                # For line drawing
                if id == 4 or id == 8:
                    points.append((cx, cy))

            # Drawing Line from top of thumb to index finger
            cv.line(frame, points[0], points[1], (255, 255, 255), 4)

            mpDraw.draw_landmarks(frame, EachHand, mpHands.HAND_CONNECTIONS)

    # fps Calculation and Drawing into the Video
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255))

    cv.imshow('Video', frame)
    cv.waitKey(10)