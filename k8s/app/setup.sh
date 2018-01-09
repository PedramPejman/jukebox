#!/bin/bash

# This script sets up the application deployment
# and service.

# Compute current directory and append secret file name to it
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROXY_SECRET="${CWD}/proxy-secret-key"

# Check if proxy secret json file exists
if [ ! -f $PROXY_SECRET ]; then
  echo "Cloudsql proxy credentials file (proxy-secret-key) not found"
  exit 1;
fi

# Check if environment variables exist
echo "Checking for required environment variables..."
if [[ -z "$DB_USERNAME" || -z "$DB_PASSWORD" ]]; 
then 
  echo "Fatal: Environment variables not set"; 
  exit 1;
fi

# Upload cloudsql credentials secrets to kubernetes
echo "Creating cloudsql database credentials secret..."
kubectl create secret generic cloudsql-db-credentials \
  --from-literal=username=$DB_USERNAME \
  --from-literal=password=$DB_PASSWORD

# Upload cloudsql instance private key secret to kubernetes
echo "Creating cloudsql instance credentials secret..."
kubectl create secret generic cloudsql-instance-credentials \
  --from-file=credentials.json=$PROXY_SECRET

# Create app deployment
echo "Creating deployment for application..."
kubectl apply -f $CWD/jb-app.yaml

# Create app service
echo "Creating the appliction service..."
kubectl apply -f $CWD/service-jb-app.yaml
