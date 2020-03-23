import event_emitter
import threading
import socket
import uuid

class SocketServer:
    
    def __init__(self, host = '0.0.0.0', port = 8082):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.connexions = {}
        self.events = event_emitter.EventEmitter()
        
    def loop(self):
        self.server.listen()
        while True:
            conn, addr = self.server.accept()
            connId = uuid.uuid4().hex
            print("Connected with address:", addr, "and with id:", connId)
            self.connexions[connId] = conn
            self.sendLine(connId, "lel")
            while True:
                data = conn.recv(32).decode("utf-8").replace("\n", "")
                if data == '': break
                self.events.emit(data, connId)
                
            print("Deconnexion")
            del self.connexions[connId]
                
            
    def start(self):
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()
        
    def sendLine(self, connId, message):
        print(len(self.connexions))
        self.connexions[connId].send((message + "\n").encode('utf8'))
        
