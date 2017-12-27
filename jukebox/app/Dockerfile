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
ENV APP_RPC_PORT 34000

# Expose port for HTTP traffic
EXPOSE ${APP_RPC_PORT}

# Set entrypoint to allow for different execution modes
ENTRYPOINT ["npm"]

# Start the server if no other arguments given
CMD [ "start" ]
