##Jukebox Application

This component is the brain of the application. It is responsible for

- Receiving RPC requests from Jukebox Frontend (jb-fe)
- Handling request data and completing the authentication flow
- Sending a series of HTTP requests to Sptify's API
- Recording raw results in the database
- Sending an RPC to Jukebox TensorFlow (jb-tf)
- Receiving the result and responding to jb-fe's original RPC

