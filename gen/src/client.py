from __future__ import print_function

import grpc

import genius_pb2 as genius
import genius_pb2_grpc as genius_grpc

def main():
    """Example client that calls the Handshake.Shake RPC method"""

    channel = grpc.insecure_channel('localhost:35000')
    stub = genius_grpc.HandshakeStub(channel)
    response = stub.Shake(genius.HandshakeRequest(name='Pedram'))
    print("Server responded with: " + response.message)

if __name__ == '__main__':
    main()
