import base64
import numpy as np
import cv2
import sys, time, math
import threading
import uuid
import os
import time
from .utils import randomString
import json
import math

class OpenCvService:
    
    def __init__(self):
        self.capture = cv2.VideoCapture(3)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 5)
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.by4dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard_create(7, 5, 0.02875, 0.023, self.dict) # 80% ratio between the marker size and square length
        self.liveVideoThreads = []
        self.saveNextFrame = False
        self.markers = False
        self.position = False
        self.calibrationPath = './calibration_data'
        self.calibrationId = ''
        self.calibrationFrameId = 0
        self.calibrationPictures = []
        self.calibrationClient = None
        self.pauseStream = False
        self.cameraConfig = {}
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.markerSize = 0.031 # in centimeters
        self.idToFind = 8
        
        self.RFlip = np.zeros((3,3), dtype=np.float32)
        self.RFlip[0,0] = 1.0
        self.RFlip[1,1] = -1.0
        self.RFlip[2,2] = -1.0
        
        self.loadCameraConfig("7adf47f3-a696-4d65-8d87-07042dab0848")

        
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def takePicture(self):
        ret, frame = self.capture.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        formatedCorners = None
        formatedIds = None
        if self.markers:
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.by4dict, parameters=self.parameters)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            formatedIds = []
            if ids is not None:
                for tag in ids:
                    formatedIds.append(int(tag[0]))
            formatedCorners = []
            if corners != None:
                for tag in corners:
                    formatedCorner = []
                    for coordinate in tag[0]:
                        formatedCorner.append([int(coordinate[0]), int(coordinate[1])])
                    formatedCorners.append(formatedCorner)

        markers = None
        position = None
        attitude = None
        dist = None
        if self.position:
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.by4dict, parameters=self.parameters, cameraMatrix=self.cameraConfig['cameraMatrix'], distCoeff=self.cameraConfig['distortionCoefficients'])
            #print(ids, corners)
            markers = []
            if ids is not None:
                formatedIds = []
                if ids is not None:
                    for tag in ids:
                        formatedIds.append(int(tag[0]))
                formatedCorners = []
                if corners != None:
                    for tag in corners:
                        formatedCorner = []
                        for coordinate in tag[0]:
                            formatedCorner.append([int(coordinate[0]), int(coordinate[1])])
                        formatedCorners.append(formatedCorner)
                for markerId in ids:
                    markerCorners = [corners[np.where(ids == markerId)[0][0]]]
                    #-- ret = [rvec, tvec, ?]
                    #-- array of rotation and position of each marker in camera frame
                    #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
                    #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
                    ret = cv2.aruco.estimatePoseSingleMarkers(markerCorners, self.markerSize, self.cameraConfig['cameraMatrix'], self.cameraConfig['distortionCoefficients'])

                    #-- Unpack the output, get only the first
                    rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                    
                    # print('rvec', rvec)
                    # print('tvec', tvec)

                    #-- Draw the detected marker and put a reference frame over it
                    cv2.aruco.drawDetectedMarkers(frame, markerCorners)
                    cv2.aruco.drawAxis(frame, self.cameraConfig['cameraMatrix'], self.cameraConfig['distortionCoefficients'], rvec, tvec, 0.08)

                    #-- Print the tag position in camera frame
                    str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
                    #print(str_position)

                    # #-- Obtain the rotation matrix tag->camera
                    # R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
                    # R_tc    = R_ct.T

                    # #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                    # roll_marker, pitch_marker, yaw_marker = self.rotationMatrixToEulerAngles(self.RFlip*R_tc)

                    # #-- Print the marker's attitude respect to camera frame
                    #str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                    #                     math.degrees(yaw_marker))
                    # print(str_attitude)
                    
                    # #-- Now get Position and attitude f the camera respect to the marker
                    # pos_camera = -R_tc*np.matrix(tvec).T

                    # str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
                    # print(str_position)

                    # #-- Get the attitude of the camera respect to the frame
                    # roll_camera, pitch_camera, yaw_camera = self.rotationMatrixToEulerAngles(self.RFlip*R_tc)
                    # str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                    #                     math.degrees(yaw_camera))
                    # print(str_attitude)
                    tvecParsed = []
                    rvecParsed = []
                    for vec in tvec:
                        tvecParsed.append(vec)
                    for vec in rvec:
                        rvecParsed.append(vec)
                    # print('tvec', tvecParsed)
                    # print('rvec', rvecParsed)
                    # print('markerId', markerId)
                    markerId = int(markerId[0])
                    markers.append({'tvec': tvecParsed, 'rvec': rvecParsed, 'id': markerId})
            
            dist = 0
            position = []
            if len(markers) == 2:
                dist = math.sqrt((markers[0]['tvec'][0] - markers[1]['tvec'][0])**2 + (markers[0]['tvec'][1] - markers[1]['tvec'][1])**2 + (markers[0]['tvec'][2] - markers[1]['tvec'][2])**2)
                for m in filter(lambda marker: marker['id'] == 42, markers):
                    originMarker = m
                for m in filter(lambda marker: marker['id'] == 8, markers):
                    mobileMarker = m
                position = [
                    mobileMarker['tvec'][0] - originMarker['tvec'][0],
                    math.sqrt((mobileMarker['tvec'][1] - originMarker['tvec'][1])**2 + (mobileMarker['tvec'][2] - originMarker['tvec'][2])**2)
                ]
                rollMarker, pitchMarker, yawMarker = self.rotationMatrixToEulerAngles(self.RFlip*(np.matrix(cv2.Rodrigues(np.array(mobileMarker['rvec']))[0]).T))
                # attitude = [
                #     math.degrees(rollMarker),
                #     math.degrees(pitchMarker),
                #     math.degrees(yawMarker)
                # ]
                attitude = math.degrees(yawMarker)
                #math.sqrt((mobileMarker['tvec'][1] - originMarker['tvec'][1])**2 + (mobileMarker['tvec'][2] - originMarker['tvec'][2])**2)
                #mobileMarker['tvec'][1] - originMarker['tvec'][1]
                #print(position)
        
        retval, buffer = cv2.imencode('.jpg', frame)
        
        data = "data:image/jpeg;base64," + str(base64.b64encode(buffer).decode("utf-8"))
        
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
        
        return [ data, {'formatedCorners': formatedCorners, 'formatedIds': formatedIds, 'markers': markers, 'dist': dist, 'position': position, 'attitude': attitude} ]
    
    def liveVideoLoop(self):
        currentThread = threading.currentThread()
        print('OPENCV: Live video ', currentThread.getName(), 'started')
        while getattr(currentThread, "doRun", True):
            while self.pauseStream:
                pass
            if not self.webSocketService.send(self.liveVideoClient, 'frame', self.takePicture()):
                setattr(currentThread, "doRun", False)
        print('OPENCV: Live video ', currentThread.getName(), 'stopped')
        
    def stopClientLiveThread(self, client):
        for thread in self.liveVideoThreads:
            print(thread[0], thread[1])
            if thread[0] == client['id']:
                print('> OPENCV: Found a live video thread to stop, with the same client id')
                thread[1].doRun = False

    def startLiveVideo(self, ws, client):
        self.webSocketService = ws
        self.liveVideoClient = client
        self.markers = False
        self.position = False
        self.stopClientLiveThread(client)
        liveVideoThread = threading.Thread(target=self.liveVideoLoop)
        self.liveVideoThreads.append((client['id'], liveVideoThread))
        liveVideoThread.start()
        
    def disableMarkers(self):
        print('> OPENCV: Marker disabled!')
        self.markers = False
        
    def enableMarkers(self):
        print('> OPENCV: Marker enabled!')
        self.markers = True
        
    def enablePosition(self):
        print('> OPENCV: Position enabled!')
        self.position = True
        
    def disablePosition(self):
        print('> OPENCV: Position disabled!')
        self.position = False
        
    def loadCameraConfig(self, calibrationId):
        f = open(self.calibrationPath + '/calibration_' + calibrationId + '/output.json')
        rawConfig = json.loads(f.read())
        self.cameraConfig = {
            'cameraMatrix': np.array(rawConfig['cameraMatrix']),
            'distortionCoefficients': np.array(rawConfig['distortionCoefficients']),
        }
        print('> OPENCV: Loaded camera config with calibration id', calibrationId)
        
    def beginCalibration(self, ws, client):
        print('> OPENCV: Calibration mode enabled! BUT NO ONE CARE!')
        # self.webSocketService = ws
        # self.calibrationClient = client
        # self.calibrationFrameId = 0
        # self.calibrationId = str(uuid.uuid4())
        # os.makedirs(os.path.abspath(self.calibrationPath + '/calibration_' + self.calibrationId))
        
    def calibrationSnapshot(self, ws, client, calibrationId = ''):
        self.webSocketService = ws
        self.calibrationClient = client
        # if the calibration id is empty or null we generate a new id
        if calibrationId == '':
            print('new calibration id')
            calibrationId = str(uuid.uuid4())
        
        self.calibrationId = calibrationId
        
        # we look if the calibration already exists, if not, we create a folder
        if 'calibration_' + calibrationId not in os.listdir(self.calibrationPath):
            print('New calibration folder!')
            os.makedirs(self.calibrationPath + '/calibration_' + calibrationId)
        # now we have a folder to hold our data
        # we retreive the highest frame id in this folder
        highestId = 0
        for frame in os.listdir(self.calibrationPath + '/calibration_' + calibrationId):
            # If this file is an jpeg image and if the frame id is higher than the current highest ist, we take this frame is as the highest id
            if '.jpg' in frame:
                frameId = int(frame.replace('.jpg', ''))
                if frameId > highestId:
                    highestId = frameId

        self.calibrationFrameId = highestId + 1
        print('> OPENCV: In calibration context, the next frame will be saved! With the frame id:', self.calibrationFrameId)
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
        self.pauseStream = True
        time.sleep(0.5)
        for path in os.listdir(self.calibrationPath + '/calibration_' + calibrationId):
            if ".jpg" in path:
                with open(self.calibrationPath + '/calibration_' + calibrationId + '/' + path, "rb") as imageFile:
                    ws.send(client, 'calibrationSave', {
                        'picture': {
                            'data': "data:image/jpeg;base64," + str(base64.b64encode(imageFile.read()).decode("utf-8")),
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
        self.pauseStream = False

    def processCalibrationData(self, ws, client, calibrationId):
        print('> OPENCV: Will process calibration data...')
        images = []
        for imagePath in os.listdir(self.calibrationPath + '/calibration_' + calibrationId):
            if ".jpg" in imagePath:
                images.append(self.calibrationPath + '/calibration_' + calibrationId + '/' + imagePath)
        print('    Images: ', images)
        allCorners, allIds, imsize = self.readChessboards(images)
        ret, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors = self.calibrateCamera(allCorners, allIds, imsize)
        # print('    ret', ret)
        # print('    camera_matrix', cameraMatrix)
        # print('    distortion_coefficients', distortionCoefficients)
        # print('    rotation_vectors', rotationVectors)
        # print('    translation_vectors', translationVectors)
        result = {
            'ret': ret,
            'cameraMatrix': [],
            'distortionCoefficients': [],
            'rotationVectors': [],
            'translationVectors': []
        }
        for matrixGroup in cameraMatrix:
            group = []
            for matrix in matrixGroup:
                group.append(matrix)
            result['cameraMatrix'].append(group)
        for coefficient in distortionCoefficients:
            result['distortionCoefficients'].append(coefficient[0])
        for vector in rotationVectors:
            #result['rotationVectors'].append({'x': vector[0][0], 'y': vector[1][0], 'z': vector[2][0]})
            result['rotationVectors'].append([vector[0][0], vector[1][0], vector[2][0]])
        for vector in translationVectors:
            #result['translationVectors'].append({'x': vector[0][0], 'y': vector[1][0], 'z': vector[2][0]})
            result['translationVectors'].append([vector[0][0], vector[1][0], vector[2][0]])
        
        jsonResult = json.dumps(result)
        print('    Result: ', jsonResult)
        
        # We save the json formated result under a output.json file in the calibration folder
        f = open(self.calibrationPath + '/calibration_' + calibrationId + '/output.json', "w")
        f.write(jsonResult + '\n')
        f.close()
        
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
        

