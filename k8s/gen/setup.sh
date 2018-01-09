#!/bin/bash

# This script sets up the genius deployment
# and service.

# Compute cwd
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\

# Create genius deployment
echo "Creating deployment for genius..."
kubectl apply -f $CWD/jb-gen.yaml

# Create genius service
echo "Creating the genius service..."
kubectl apply -f $CWD/service-jb-gen.yaml
