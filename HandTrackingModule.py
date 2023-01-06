import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True): 
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):  
                    if draw:               
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, landmarkID = 0, draw = True):
        lmmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmmList.append([id, cx, cy])
                if draw and id ==landmarkID:
                    cv.circle(img, (cx, cy), 10, (0,255,0), cv.FILLED)      
        return lmmList


def main():
    cap = cv.VideoCapture(0)
    prev_frame_time = 0
    new_frame_time = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, landmarkID=4)
        if(len(lmList)!=0):
            print(lmList[4])
        new_frame_time = time.time()

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        cv.putText(img, str(fps), (10,70), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
        cv.imshow('Image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            


if __name__ == "__main__":
    main()
