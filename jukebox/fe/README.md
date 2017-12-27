##Jukebox Frontend

This component is responsible for:

- Receiving the initial HTTP request from the client
- Forwarding it to the Spotify API
- Extracting the client code and sending an RPC to Jukebox Application (jb-app)
- Receiving the RPC response
- Responding to the client with the final interactive HTML page
