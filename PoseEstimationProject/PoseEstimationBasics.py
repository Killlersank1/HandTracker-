import time
import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture('vids/video.mp4')
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
pTime = 0

while True:
    isTrue, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 50), cv.FONT_ITALIC, 2.0, (255, 255, 0), 3)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id, cx, cy])
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


    cv.imshow( 'Video', img)
    cv.waitKey(1)