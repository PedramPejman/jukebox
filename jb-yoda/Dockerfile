# Use java 8 base
FROM java:8

# Update package manager and install maven
RUN apt-get update
RUN apt-get install -yqq maven

# Create and set working directory
ENV YODA_ROOT /usr/jukebox/yoda
RUN mkdir -p $YODA_ROOT
WORKDIR $YODA_ROOT

# Copy in container resources
COPY . $YODA_ROOT

# Set environment variables
ENV YODA_RPC_PORT 37000
ENV GCP_PROJECT_ID pedram-demo-0
ENV GOOGLE_APPLICATION_CREDENTIALS $YODA_ROOT/yoda-dev-secret.json
ENV PUBSUB_TOPIC_JUKEBOX_CREATED jukebox-created
ENV PUBSUB_SUB_JUKEBOX_CREATED_YODA jukebox-created-yoda

# Build/verify mvn project
RUN ["mvn", "verify"]
RUN ["mvn", "package"]

# Expose port for gRPC requests
EXPOSE $YODA_RPC_PORT

# Make default entrypoint start Yoda's gRPC server
ENTRYPOINT ["mvn", "exec:java", "-Dexec.mainClass=life.jukebox.yoda.YodaSubscriberClient"]
