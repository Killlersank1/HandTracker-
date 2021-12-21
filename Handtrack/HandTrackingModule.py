import cv2 as cv
import mediapipe as mp
import time


class handDetector:

    def __init__(self,mode = False, maxHands = 2, detectionConf = 0.5 , trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hand = self.mpHands.Hands(self.mode,self.maxHands,
                                       self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw = True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for EachHand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, EachHand, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self,img ,Handno = 0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            MyHand = self.results.multi_hand_landmarks[Handno]
            for id , lm in enumerate(MyHand.landmark):
                h ,w , c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img , (cx,cy),5,(255,0,255),-1)
        return lmlist





def main():
    cTime = 0
    pTime = 0
    detector = handDetector()
    cap = cv.VideoCapture(0)
    while True:
        # For Reading the Video
        isTrue, frame = cap.read()
        frame = detector.findHands(frame)
        lmlist = detector.findPosition(frame)
        if len(lmlist) != 0:
            print(lmlist[6])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(frame, str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255))

        cv.imshow('Video', frame)
        cv.waitKey(10)


if __name__ == '__main__':
    main()