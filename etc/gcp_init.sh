#!/bin/bash

# Install beta components
gcloud components update
gcloud components install beta

# Create Pub/Sub topics and subscriptions
gcloud beta pubsub topics create jukebox-created
gcloud beta pubsub subscriptions create jukebox-created-yoda --topic jukebox-created
