#!/bin/bash

# Delete genius service
echo "Deleting genius service..."
kubectl delete -f service-jb-gen.yaml 

# Delete genius 
echo "Deleting genius deployment..."
kubectl delete -f jb-gen.yaml
