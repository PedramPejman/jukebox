#!/bin/bash

# Delete secrets
echo "Deleting secrets..."
kubectl delete secret \
  cloudsql-db-credentials \
  cloudsql-instance-credentials

# Delete application service
echo "Deleting application service..."
kubectl delete -f service-jb-app.yaml 

# Delete application
echo "Deleting application deployment..."
kubectl delete -f jb-app.yaml
