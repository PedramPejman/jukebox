#!/bin/bash

# At some point, this has to become the source of truth for all environment variables that
# different Jukebox services need to consume. Until then, this file will track all cross-service vars.
# TODO(pedrampejman): Understand the right way to inject environment variables from host
# to the container at build time.

# Set GCP project name
export GCP_PROJECT_ID=pedram-demo-0

# Set Pub/Sub topic and subscriptions
export PUBSUB_TOPIC_JUKEBOX_CREATED=jukebox-created
export PUBSUB_SUB_JUKEBOX_CREATED_YODA=jukebox-created-yoda
