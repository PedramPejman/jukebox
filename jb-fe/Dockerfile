# Use stable base node image 
FROM node:7-onbuild

# Create app directory
ENV FE_ROOT /usr/jukebox/jb-fe
RUN mkdir -p $FE_ROOT
WORKDIR $FE_ROOT

# Bundle app resources
COPY . $FE_ROOT

# Install app dependencies
RUN npm install

# The port that FE container should listen on
ENV FE_HTTP_PORT 8080

# TODO: THIS NEEDS TO BE MOVED TO KB SECRETS
ENV CLIENT_SECRET bf1237ed525746b5bde998ec320c2a0f

# Expose port for HTTP traffic
EXPOSE ${FE_HTTP_PORT}

# Start the server
CMD [ "npm", "start" ]
