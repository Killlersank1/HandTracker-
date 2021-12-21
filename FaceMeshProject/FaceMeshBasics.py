import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)

mpMesh = mp.solutions.face_mesh
mesh = mpMesh.FaceMesh(False)

mpDraw = mp.solutions.drawing_utils


while True:
    isTrue, frame = cap.read()

    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for id, face_landmark in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(frame, face_landmark, mpMesh.FACEMESH_CONTOURS)
            print(id, face_landmark)

    cv.imshow('Vid', frame)
    cv.waitKey(1)