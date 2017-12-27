#!/bin/bash

# Make sure we are running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# Make sure the docker bridge network 'jukebox-net' exists
net_out=$(docker network list | grep jukebox-net)
if [ -z "$net_out" ]; then
  echo "Creating docker network 'jukebox-net'"
  docker network create jukebox-net
else
  echo "docker network 'jukebox-net' already exists"
fi
