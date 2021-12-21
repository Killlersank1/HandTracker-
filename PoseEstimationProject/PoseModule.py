import time
import cv2 as cv
import mediapipe as mp

class poseDetection:
    def __init__(self, mode =False, complexity=1, smooth=True, detectConf=0.5, trackConf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity,  self.smooth, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:

            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        lmlist = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([idx, cx, cy])
        return lmlist


def main():
    cap = cv.VideoCapture('vids/video.mp4')
    pTime = 0
    detector = poseDetection()
    while True:
        isTrue, img = cap.read()
        img = detector.findPose(img)
        llist = detector.getPosition(img)
        if len(llist) != 0:
            print(llist[14])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 50), cv.FONT_ITALIC, 2.0, (255, 255, 0), 3)

        cv.imshow('Video', img)
        cv.waitKey(1)


if __name__ == '__main__':
    main()
