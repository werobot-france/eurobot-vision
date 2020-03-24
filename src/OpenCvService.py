import base64
import numpy as np
import cv2
import sys, time, math
import threading
import uuid
import os
import time

class OpenCvService:
    
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 5)
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.board = cv2.aruco.CharucoBoard_create(7, 5, 1, .8, self.dict)
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
        
        data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        
        if self.saveNextFrame:
            path = os.path.abspath(self.calibrationPath + '/calibration_' + self.calibrationId + '/' + str(self.calibrationFrameId) + '.jpg')
            cv2.imwrite(path, frame)
            self.saveNextFrame = False
            print('> OPENCV: Frame saved under', path)
            self.calibrationPictures.append(path)
            snapshotObject = {
                'calibrationId': self.calibrationId,
                'calibrationFrameId': self.calibrationFrameId,
                'format': 'jpg',
                'path': path,
                'data': data,
                'calibrationPath': self.calibrationPath
            }
            self.webSocketService.send(self.calibrationClient, 'calibrationSnapshot', snapshotObject)
        
        return data
    
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
        os.makedirs(os.path.abspath(self.calibrationPath + '/calibration_' + self.calibrationId))
        
    def calibrationSnapshot(self):
        self.calibrationFrameId = self.calibrationFrameId + 1
        print('> OPENCV: In calibration context, the next frame will be saved!', self.calibrationFrameId)
        self.saveNextFrame = True
        
    def deleteInputData(self, calibrationId, calibrationFrameId):
        os.unlink(os.path.abspath(self.calibrationPath + '/calibration_' + calibrationId + '/' + calibrationFrameId + '.jpg'))
        
    def fetchSaves(self, ws, client):
        saves = []
        for path in os.listdir(self.calibrationPath):
            if "calibration_" in path:
                saves.append({
                    'id': path.replace('calibration_', ''),
                    'path': path,
                    'count': len(os.listdir(self.calibrationPath + '/' + path))
                })
        ws.send(client, 'calibrationSaves', {
            'saves': saves
        })

    def fetchSave(self, ws, client, calibrationId):
        pictures = []
        for path in os.listdir(self.calibrationPath + '/calibration_' + calibrationId):
            if ".jpg" in path:
                with open(self.calibrationPath + '/calibration_' + calibrationId + '/' + path, "rb") as imageFile:
                    ws.send(client, 'calibrationSave', {
                        'picture': {
                            'data': "data:image/jpeg;base64," + base64.b64encode(imageFile.read()).decode("utf-8"),
                            'path': path,
                            'calibrationFrameId': path.replace('.jpg', ''),
                            'calibrationId': calibrationId
                        }
                    })
                    print('frame ' + calibrationId + ' sent')
                    time.sleep(0.2)
        
        print('Frame id after load:', self.calibrationFrameId)
        self.calibrationFrameId = len(pictures) + 1
        self.calibrationId = calibrationId

    def processCalibrationData(self, ws, client, calibrationId):
        print('> OPENCV: Will process calibration data...')
        images = []
        for imagePath in os.listdir(self.calibrationPath + '/calibration_' + calibrationId):
            if ".jpg" in imagePath:
                images.append(self.calibrationPath + '/calibration_' + calibrationId + '/' + imagePath)
        print('    Images: ', images)
        allCorners, allIds, imsize = self.readChessboards(images)
        ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = self.calibrateCamera(allCorners, allIds, imsize)
        print('    ret', ret)
        print('    camera_matrix', camera_matrix)
        print('    distortion_coefficients0', distortion_coefficients0)
        print('    rotation_vectors', rotation_vectors)
        print('    translation_vectors', translation_vectors)
        result = {
            'ret': ret,
            'cameraMatrix': camera_matrix,
            'distortion_coefficients0': distortion_coefficients0,
            'rotation_vectors': rotation_vectors,
            'translation_vectors': translation_vectors
        }
        ws.send(client, 'calibrationOutput', result)
    
    def readChessboards(self, images):
        """
        Charuco base pose estimation.
        """
        print("> OPENCV: Charuco base estimation...")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in images:
            print("    => Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.dict)

            if len(corners)>0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                    winSize = (3,3),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,self.board)
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator+=1

        imsize = gray.shape
        return allCorners, allIds, imsize
    
    def calibrateCamera(self, allCorners, allIds, imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("> OPENCV: Camera calibration...")

        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        #flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=self.board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
        

