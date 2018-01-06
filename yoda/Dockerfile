# Use java 8 base
FROM java:8

# Update package manager and install maven
RUN apt-get update
RUN apt-get install -yqq maven

# Create and set working directory
RUN mkdir -p /usr/jukebox/yoda
WORKDIR /usr/jukebox/yoda

# Copy project object model and source directory
COPY pom.xml .
COPY src ./src
COPY protos ./protos

# Build/verify mvn project
RUN ["mvn", "verify"]
RUN ["mvn", "package"]

# Make default entrypoint start Yoda's gRPC server
ENTRYPOINT ["mvn", "exec:java", "-Dexec.mainClass=life.jukebox.yoda.YodaGrpcServer"]