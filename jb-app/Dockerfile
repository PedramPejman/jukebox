# Use stable base node image 
FROM node:7-onbuild

# Create jb-app directory
ENV APP_ROOT /usr/jukebox/jb-app
RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

# Bundle app resources
COPY . $APP_ROOT

# Install app dependencies
RUN npm install

# Set environment variables
ENV APP_RPC_PORT 34000
ENV GOOGLE_APPLICATION_CREDENTIALS $APP_ROOT/app-dev-secret.json
ENV PUBSUB_TOPIC_JUKEBOX_CREATED jukebox-created

# Expose port for gRPC traffic
EXPOSE $APP_RPC_PORT

# Set entrypoint to allow for different execution modes
ENTRYPOINT ["npm"]

# Start the server if no other arguments given
CMD [ "start" ]
