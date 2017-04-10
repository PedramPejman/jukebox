#!/bin/bash

# This script sets up the genius deployment
# and service.

# Create genius deployment
echo "Creating deployment for genius..."
kubectl apply -f jb-gen.yaml

# Create genius service
echo "Creating the genius service..."
kubectl apply -f service-jb-gen.yaml
