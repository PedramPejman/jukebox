# Use stable base node image 
FROM node:7-onbuild

# Create app directory
RUN mkdir -p /usr/jukebox/app
WORKDIR /usr/jukebox/app

# Bundle app resources
COPY . /usr/jukebox/app

# Install app dependencies
RUN npm install

# Set environment variables
ENV APP_RPC_HOST 0.0.0.0
ENV APP_RPC_PORT 34000
ENV GEN_RPC_HOST jb-gen
ENV GEN_RPC_PORT 35000 

# Expose port for HTTP traffic
EXPOSE ${APP_RPC_PORT}

# Start the server
CMD [ "node", "./src/server.js" ]
