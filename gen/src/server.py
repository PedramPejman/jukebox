from concurrent import futures
import time
import os

import grpc

import genius_pb2 as genius
import genius_pb2_grpc as genius_grpc

ONE_DAY = 60 * 60 * 24
GEN_RPC_PORT = os.getenv('GEN_RPC_PORT', '35000')

class Handshake(genius_grpc.HandshakeServicer):
    """Implementation of Handshake RPC server"""

    def Shake(self, request, context):
        """Implements the Shake test service RPC handler"""

        print("Received HandshakeRequest with name %s" %(request.name))
        return genius.HandshakeResponse(message="Hello, %s!" % request.name)

def main():
    """Sets up the server and listens on GEN_RPC_PORT"""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    genius_grpc.add_HandshakeServicer_to_server(Handshake(), server)
    server.add_insecure_port("[::]:" + GEN_RPC_PORT)
    server.start()
    print("Server starting on port %s" %(GEN_RPC_PORT))
    
    try:
        while True:
            time.sleep(ONE_DAY)
    except KeyboardInterrupt:
        server.stop(0)

if (__name__ == "__main__"):
    main()
