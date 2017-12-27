FROM java:8

RUN apt-get update
RUN apt-get install -yqq maven

RUN mkdir -p /usr/jukebox/yoda
WORKDIR /usr/jukebox/yoda

COPY . /usr/jukebox/yoda

RUN ["mvn", "dependency:resolve"]
RUN ["mvn", "verify"]
RUN ["mvn", "package"]

ENTRYPOINT ["java", "-cp", "target/yoda-1.0-SNAPSHOT.jar", "life.jukebox.App"]

ENV YOD_RPC_PORT 34000

EXPOSE ${YOD_RPC_PORT}

