# From lightweight base python image
FROM python:latest

# Create app directory
ENV GEN_ROOT /usr/jukebox/gen
RUN mkdir -p $GEN_ROOT
WORKDIR $GEN_ROOT

# Bundle app resources
COPY . $GEN_ROOT

# Install dependencies
RUN pip install -r requirements.txt

# Compile proto files into Python
RUN python3 -m grpc_tools.protoc -I ./protos --python_out=./src/protos --grpc_python_out=./src/protos ./protos/genius.proto

# Set environment variables
ENV GEN_RPC_HOST "0.0.0.0"
ENV GEN_RPC_PORT 35000

# Expose port for RPC traffic
EXPOSE ${GEN_RPC_PORT}

# Start the server
CMD [ "python", "./src/server.py" ]
