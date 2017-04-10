#!/bin/bash

# This script sets up the frontned deployment
# and service.

# Create the frontend deployment
echo "Creating deployment for frontend..."
kubectl apply -f jb-fe.yaml

# Create frontend service
echo "Creating the frontend service..."
kubectl apply -f service-jb-fe.yaml
