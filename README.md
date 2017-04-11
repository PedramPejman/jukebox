# Jukebox
[![Build Status](https://travis-ci.org/PedramPejman/jukebox.svg?branch=master)](https://travis-ci.org/PedramPejman/jukebox)

## Purpose

This project has two main purposes

- Act as a playground for the authors, in which they can test and experiment with relatively-new technologies such as kubernetes, docker, grpc, protobuf, etc. under a service-oriented architecture. 
- Build a dope jukebox using Spotify's API

## Architecture Overview
This application is divided into four different services:

- Frontend (FE), which acts as the presentation layer
- Application (APP), which acts as the brain, delegating computation to GEN and supplying FE with content
- Genius (GEN), which acts as the computation engine, executing all ML routines
- Mysql database, which acts as the primary agent for persistent storage

## Production Services
Jukebox is currently in alpha, hosted on a Google Container Engine cluster. 

Here are the addresses for the live production services and what ports they expect what type of traffic on:


| Service        | Service ID           | Protocol | IP              | Port  |
|:--------------:|:--------------------:|:--------:|:---------------:|------:|
| Frontend       | jb-fe-prod           | HTTP     | 104.196.62.203  | 80    |
| Application    | jb-app-prod          | gRPC     | 35.185.41.242   | 34000 |
| Genius         | jb-gen-prod          | gRPC     | 104.196.209.40  | 35000 |
| MySql          | jb-sql-dev           | TCP      | 104.196.23.18   | 3306  |

## Installation

Although pretty intuitive, let's go over the installation process for each service. On a high level, there are two types of installations: single-service and multi-service. By this, I mean you may either want to run a single service locally (for example FE) or run multiple services locally on your machine (for example FE and APP). You can accomplish either objective with or without Docker, so let's go over it.

### With Docker
TODO: Need to amend Dockerfile to add DNS record for jb-app so that it works outside Kubernetes

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

- A user visits the site's root page 
- FE receives the request
- FE leads user through the Spotify OAuth procedure 
- FE retrieves the necessary information to make API requests on behalf of the user (secret and refresh tokens) 
- FE sends these tokens to app via an RPC call
- FE renders spinner for user
- FE blocks and waits for a RPC response

- APP receives RPC
- APP makes a GET request to Spotify's API on /user/top/ 
- APP logs response in the database 
- APP makes a GET request to Spotify's API on /audio-features?ids={track-id+}
- APP computes default averages for audio features (jukebox parameters)
- APP responds to RPC with top tracks and list of parameters and their default values  

- FE receives RPC response and forwards user to /configure
- The user chooses seed tracks and adjusts playlist parameters
- The user triggers the CTA 
- FE sends RPC to APP with list of seed tracks and final parameter values

- APP receives the request
- APP makes a GET request to Spotify's API on /recommendations
- APP logs results in databse
- APP makes a PUT request to Spotify's API on /users/{user_id}/playlists
- APP responds to the RPC with the playlist's URI

- FE receives RPC response and forwards user to /play 
- The user is presented with a Spotify playlist widget containing their jukebox playlist

## Contributions
To contribute to this project (whether you are one of the original authors or not) do the following.
- Follow installation instrutions
- Do all your development on a separate branch (with a descriptive name)
- Submit a pull request
- Respond to any comments and make necessary changes
- Sit back and relax as your code changes the world :)
