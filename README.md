# Jukebox
[![Build Status](https://travis-ci.org/PedramPejman/jukebox.svg?branch=master)](https://travis-ci.org/PedramPejman/jukebox)

## Purpose

This project has two main purposes

- Act as a playground for the authors, in which they can test and experiment with relatively-new technologies such as kubernetes, docker, grpc, protobuf, etc. under a service-oriented architecture. 
- Build a dope playlist generator using Spotify's API

## Architecture Overview
This application is divided into four different services:

- Frontend (FE), which acts as the presentation layer and web server
- Application (APP), which acts as the brain, delegating heavy computation to GEN and supplying FE with content
- Genius (GEN), which acts as the computation engine, executing all ML routines
- Knowledge Graph (KG), a Bigtable instance which acts as Genius' persistent layer, tracking information about users' music taste
- Database (DB), a Mysql database which acts as Application's persistent layer, tracking sensitive and crucial data


## Continuous Deployment
The Jukebox project adheres to principles of continuous deployment. All successful merges into the *master branch* are automatically pushed to [the staging envrionment](http://staging.jukebox.life) and merges into *production branch* automatically pushed to [the production environment](http://jukebox.life). Support for canarying will be added when the beta is released and a fully automated A/B test pipeline is imperative.

Jukebox is currently in alpha, hosted on a Google Container Engine cluster; here are the addresses for the live production services and what ports they expect what type of traffic on:


| Service         | Service ID           | Protocol | IP              | Port  |
|:---------------:|:--------------------:|:--------:|:---------------:|------:|
| Frontend        | jb-fe-prod           | HTTP     | 104.196.62.203  | 80    |
| Application     | jb-app-prod          | gRPC     | 35.185.41.242   | 34000 |
| Genius          | jb-gen-prod          | gRPC     | 104.196.209.40  | 35000 |
| Knowledge Graph | jb-kg-prod           | gRPC     | 104.113.207.32  | 36000 |
| Database        | jb-sql-dev           | TCP      | 104.196.23.18   | 3306  |

## Installation

Although it's pretty intuitive, let's go over the installation process for each service. On a high level, there are two types of installations: single-service and multi-service. By this, I mean you may either want to run a single service locally (for example FE) or run multiple services locally on your machine (for example FE and APP). You can accomplish either objective with or without Docker, so let's go over it.

### With Docker
In order to run all services locally, you can use the following commands:

```{r, engine='bash', count_lines}
# Clone the repo
git clone https://github.com/pedrampejman/jukebox.git

# Build protos for FE and APP
cd jukebox
make build-fe && make build-app

# Run the FE container
docker run --rm -it --net=host -v $(pwd)/fe/src:/usr/jukebox/fe/src pedrampejman/jb-fe

# Run the APP container
docker run --rm -it --net=host -v $(pwd)/app/src:/usr/jukebox/app/src pedrampejman/jb-app

# Run the GEN container
docker run --rm -it --net=host -v $(pwd)/gen/src:/usr/jukebox/gen/src pedrampejman/jb-gen
``` 

Notice a number of nuances (if you're not super familiar with Docker magic). First, the ```--rm``` flag conveniently deletes the container once we're done with it. Second, with the ``--network`` option, we're specifying that we'd like to run on the host machine's network stack; meaning, all the network interfaces defined on the host will be accesible to the container (i.e. this will let the container service a visit to localhost:8080 on the host machine). Lastly, with the ```--volume``` option, we're mapping our ```src``` directory to the container's ```src``` directory. Without this, every time you made changes to the source code, you'd have to build the service before running it, which may take up to a couple minutes. With this option, you can make changes locally, and simply run the container with the updated code in seconds, thus greatly speeding up the development cycle.

Now, if you visit http://localhost:8080 on your browser, you should see everything working!

### Without Docker
Let's go over setting up Frontend as it is the easiest.

### Frontend

To run FE locally, simply do the following

- Clone this repository
- Build the specific service using make
- Point it to the appropriate endpoints
- Install dependencies using 'npm install'
- Run your service using 'npm start'

If you're confused by the third step read this: when you run a service locally, you must tell it where to look for the other services. There are usually 2 mechanisms for doing this 1) environment variables 2) command-line arguments. I have chosen method 1, which is why we export host addresses as environment variables.

This is what the process should roughly look like.

```{r, engine='bash', count_lines}
# Clone the repo
git clone https://github.com/pedrampejman/jukebox.git

# Build the frontend service
cd jukebox
make build-fe

# Point FE to production APP service
export APP_RPC_HOST=35.185.41.242

# Install dependencies
cd fe
npm install

# Run service 
npm start
``` 

Now if you visit localhost:8080 on your machine (or inside your container if you're running the service using the Docker image) you'll see that your FE service is communicating with the live APP service. 

## Communication Pattern Deep-Dive
Since this is the first time I have designed a distributed application, I have kept all communication patterns simple. The APP service is also a bit too heavy and should be broken up into more modular services but again, I'm new to this space and wanted to just get my toes wet, while still gaining practice architecting an app following SOA principles. The communication pattern can be best understood through the use of an example. 

TODO: add graphic outlining the communication pattern

## Contributions
To contribute to this project (whether you are one of the original authors or not) do the following.
- Follow installation instrutions
- Do all your development on a separate branch (with a descriptive name)
- Submit a pull request
- Respond to any comments and make necessary changes
- Sit back and relax as your code changes the world :)

Don't hesitate to reach out to me personally if you'd like to contribute but unsure where to start.

## Credits
Regardless of what Jukebox becomes in the future, it was started as a school project with [Mira Holford](https://www.linkedin.com/in/miraholford/), [Jessica Zimmerman](https://www.linkedin.com/in/jessica-zimmerman-3342b4a2/) and Nathan Nayda. [Hoyle Wang](https://www.linkedin.com/in/hoylewang/) was instrumental in the product definition process and helped shape core aspects of the user experience. 

I will do my best to keep this list updated, crediting those who contribute to the project. Jukebox belongs to the opensource community and for as long as I have a say in its direction, will never adopt a monetization strategy whose objective is anything but keeping the project alive.
