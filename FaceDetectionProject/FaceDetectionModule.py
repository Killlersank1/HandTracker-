import cv2 as cv
import mediapipe as mp
import time

class FaceDetection:
    def __init__(self, detect_conf=0.5, select=0):
        self.detect_conf = detect_conf
        self.select = select
        self.mpface = mp.solutions.face_detection
        self.face = self.mpface.FaceDetection(self.detect_conf, self.select)
        self.mpdraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.face.process(img_rgb)
        # print(results.detections)

        if self.results.detections:
            for idx, detection in enumerate(self.results.detections):
                # mpdraw.draw_detection(img, detection)
                # print(detection.location_data.relative_bounding_box)
                ih, iw, ic = img.shape
                bbox_c = detection.location_data.relative_bounding_box
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih),\
                        int(bbox_c.width * iw), int(bbox_c.height * ih)
                if draw:
                    cv.rectangle(img, bbox, (255, 0, 0), 3)
                    cv.putText(img, str(int(detection.score[0] * 100)), (bbox[0], bbox[1] - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 2)
        return img

def main():
    cap = cv.VideoCapture(0)
    detector = FaceDetection()
    ptime = 0
    while True:
        success, img = cap.read()

        img = detector.findFace(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(img, str(int(fps)), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 3)

        cv.imshow('Video', img)
        cv.waitKey(1)


if __name__ == '__main__':
    main()
