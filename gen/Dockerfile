# From base image containing Python3 scikit-learn 
# with Jupyter docker image based on alpine
FROM smizy/scikit-learn:latest

# Create app directory
RUN mkdir -p /usr/jukebox/gen
WORKDIR /usr/jukebox/gen

# Bundle app resources
COPY . /usr/jukebox/gen

# Set environment variables
ENV APP_RPC_HOST jb-app
ENV APP_RPC_PORT 34000
ENV GEN_RPC_HOST "0.0.0.0"
ENV GEN_RPC_PORT 35000

# Expose port for RPC traffic
EXPOSE ${GEN_RPC_PORT}

# Start the server
CMD [ "python3", "./src/main.py" ]
