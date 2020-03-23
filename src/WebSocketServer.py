from websocket_server import WebsocketServer
from .OpenCvService import OpenCvService
import logging
import base64
import json

class WebSocketServer:
    
    def __init__(self):
        self.server = WebsocketServer(8082, host='0.0.0.0')
        self.server.set_fn_new_client(self.onClient)
        self.server.set_fn_message_received(self.onMessage)
        self.openCvService = OpenCvService()
    
    def onClient(self, client, server):
        print("new client", client)
    
    def onMessage(self, client, server, message):
        print("new message", message)
        messageParsed = json.loads(message)
        command = messageParsed['command']
        args = messageParsed['args']
        if command == "ping":
            print('PONG!')
        if command == "photo":
            print('TAKE PHOTO')
            self.send(client, "PIC#" + self.openCvService.takePicture())
        if command == "begin_calibration":
            # will clean the calib_tmp folder
            self.openCvService.beginCalibration(self, client)
        if command == "calibration_snapshot":
            self.openCvService.calibrationSnapshot()
        if command == "enable_markers":
            # enable marker for this specific client
            self.openCvService.enableMarkers()
        if command == "disable_markers":
            # enable marker for this specific client
            self.openCvService.disableMarkers()
        if command == "live":
            # create a new thread
            self.openCvService.startLiveVideo(self, client)
        if command == "calibration_get_input_data":
            with open(args['path'], "rb") as imageFile:
                sent = "data:image/jpeg;base64," + base64.b64encode(imageFile.read()).decode("utf-8")
                self.send(client, "calibrationData", sent)

    def send(self, client, responseType, data = None):
        self.server.send_message(client, json.dumps({'responseType': responseType, 'data': data}))

    def start(self):
        self.server.run_forever()
