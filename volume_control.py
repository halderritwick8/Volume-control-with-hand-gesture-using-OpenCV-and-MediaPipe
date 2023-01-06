import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule as htm
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
detector = htm.handDetector(detectionCon=0.7)
vol = 0
volBar = 375
volPerc = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()

minVol = volrange[0]
maxVol = volrange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if(len(lmList)!=0):
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv.circle(img, (x1,y1), 10, (255,0,255), cv.FILLED)
        cv.circle(img, (x2,y2), 10, (255,0,255), cv.FILLED)
        cv.circle(img, (cx, cy), 10, (255,0,255), cv.FILLED)
        cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # gesture length --> 20 - 350
        # volume range --> -65 - 0

        vol = np.interp(length, [50, 160], [minVol, maxVol])
        volBar = np.interp(length, [50, 160], [375, 200])
        volPerc = np.interp(length, [50, 160], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

    cv.rectangle(img, (20, 200), (50, 375), (0,0,255), 3)
    cv.rectangle(img, (20, int(volBar)), (50, 375), (0,255,255), cv.FILLED)
    cv.putText(img, f'{int(volPerc)}%', (15,410), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)



    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    cv.putText(img, f'fps : {str(fps)}', (5,30), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv.imshow('Frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
