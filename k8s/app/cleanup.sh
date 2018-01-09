#!/bin/bash

# Compute cwd
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\

# Delete secrets
echo "Deleting secrets..."
kubectl delete secret \
  cloudsql-db-credentials \
  cloudsql-instance-credentials

# Delete application service
echo "Deleting application service..."
kubectl delete -f $CWD/service-jb-app.yaml 

# Delete application
echo "Deleting application deployment..."
kubectl delete -f $CWD/jb-app.yaml
