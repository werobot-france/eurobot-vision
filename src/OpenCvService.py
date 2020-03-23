import base64
import numpy as np
import cv2
import sys, time, math
import threading
import uuid
import os

class OpenCvService:
    
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 5)
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.liveVideoThreads = []
        self.saveNextFrame = False
        self.markers = False
        self.calibrationPath = './calibration_data'
        self.calibrationId = ''
        self.calibrationFrameId = 0
        self.calibrationPictures = []
        self.calibrationClient = None

    def takePicture(self):
        ret, frame = self.capture.read()
        parameters = cv2.aruco.DetectorParameters_create()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.markers:
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.dict, parameters=parameters)
            cv2.aruco.drawDetectedMarkers(frame, corners)
        
        retval, buffer = cv2.imencode('.jpg', frame)
        
        if self.saveNextFrame:
            path = os.path.abspath(self.calibrationPath + '/' + self.calibrationId + '/' + str(self.calibrationFrameId) + '.jpg')
            cv2.imwrite(path, frame)
            self.saveNextFrame = False
            print('> OPENCV: Frame saved under', path)
            self.calibrationPictures.append(path)
            snapshotObject = {
                'calibrationId': self.calibrationId,
                'calibrationFrameId': self.calibrationFrameId,
                'format': 'jpg',
                'path': path,
                'calibrationPath': self.calibrationPath
            }
            self.webSocketService.send(self.calibrationClient, 'calibrationSnapshot', snapshotObject)
        
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
    
    def liveVideoLoop(self):
        currentThread = threading.currentThread()
        print('OPENCV: Live video ', currentThread.getName(), 'started')
        while getattr(currentThread, "doRun", True):
            self.webSocketService.send(self.liveVideoClient, 'frame', self.takePicture())
        print('OPENCV: Live video ', currentThread.getName(), 'stopped')

    def startLiveVideo(self, ws, client):
        self.webSocketService = ws
        self.liveVideoClient = client
        self.markers = False
        for thread in self.liveVideoThreads:
            print(thread[0], thread[1])
            if thread[0] == client['id']:
                print('> OPENCV: Found a live video thread to stop, with the same client id')
                thread[1].doRun = False
        
        liveVideoThread = threading.Thread(target=self.liveVideoLoop)
        self.liveVideoThreads.append((client['id'], liveVideoThread))
        liveVideoThread.start()
        
    def disableMarkers(self):
        print('> OPENCV: Marker disabled!')
        self.markers = False
        
    def enableMarkers(self):
        print('> OPENCV: Marker enabled!')
        self.markers = True
        
    def beginCalibration(self, ws, client):
        print('> OPENCV: Calibration mode enabled!')
        self.webSocketService = ws
        self.calibrationClient = client
        self.calibrationFrameId = 0
        self.calibrationId = str(uuid.uuid4())
        os.makedirs(os.path.abspath(self.calibrationPath + '/' + self.calibrationId))
        
    def calibrationSnapshot(self):
        print('> OPENCV: In calibration context, the next frame will be saved!')
        self.calibrationFrameId = self.calibrationFrameId + 1
        self.saveNextFrame = True

    def calibrate(self):
        pass
    
    def saveCalibrationSnapshot(self):
        pass

