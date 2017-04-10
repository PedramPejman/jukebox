#!/bin/bash

# Delete frontend service
echo "Deleting frontend service..."
kubectl delete -f service-jb-fe.yaml 

# Delete frontend
echo "Deleting frontend deployment..."
kubectl delete -f jb-fe.yaml
