import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpface = mp.solutions.face_detection
face = mpface.FaceDetection()

mpdraw = mp.solutions.drawing_utils

ptime = 0
while True:
    success, img = cap.read()

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv.putText(img, str(int(fps)), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 3)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = face.process(img_rgb)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpdraw.draw_detection(img, detection)
            # print(detection.location_data.relative_bounding_box)
            ih, iw, ic = img.shape
            bbox_c = detection.location_data.relative_bounding_box
            bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih),\
                   int(bbox_c.width * iw), int(bbox_c.height * ih)
            cv.rectangle(img, bbox, (255, 0, 0), 2)
            cv.putText(img, str(int(detection.score[0] *100)), (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 3)

    cv.imshow('Video', img)
    cv.waitKey(1)
