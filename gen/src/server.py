from concurrent import futures
import time
import os
import sys

# Fix dealing with Protobuf issue #1491: Add protos directory to PYTHONPATH
PROTOS_DIR = "protos"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), PROTOS_DIR))

import grpc

import protos.genius_pb2 as genius_pb
import protos.genius_pb2_grpc as genius_grpc

from genius import *

ONE_DAY = 60 * 60 * 24
GEN_RPC_PORT = os.getenv('JB_GEN_SERVICE_PORT', '35000')

class Genius(genius_grpc.GeniusServicer):
    """ Implementation of Genius RPC server """

    def Shake(self, request, context):
        """ Implements the Shake test RPC method"""

        print("Received HandshakeRequest with name %s" %(request.name))
        return genius_pb.HandshakeResponse(message="Hello, %s!" % request.name)

    def GetSeedTracks(self, request, context):
        """ Implements the GetSeedTracks RPC handler"""

        print("Received GetSeedTracks with access token %s" %(request.authCreds.accessToken))
        raw_tracks = get_seed_tracks(request.authCreds.accessToken)
        tracks = []
        for track in raw_tracks:
            
            # Create list of artists
            artists = []
            for artist in track['artists']:
                artists.append(genius_pb.Artist(
                    id = artist['id'],
                    name = artist['name']))
            
            # Append track to the list of tracks
            tracks.append(genius_pb.Track(
              id=track['id'], 
              name=track['name'],
              album=track['album']['name'],
              imageUri=track['album']['images'][0]['url'],
              artists=artists))

        return genius_pb.GetSeedTracksResponse(seedTracks=tracks)

def main():
    """ Sets up the server and listens on GEN_RPC_PORT """

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    genius_grpc.add_GeniusServicer_to_server(Genius(), server)

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
