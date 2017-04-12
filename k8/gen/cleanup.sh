#!/bin/bash

# Compute cwd
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\

# Delete genius service
echo "Deleting genius service..."
kubectl delete -f $CWD/service-jb-gen.yaml 

# Delete genius 
echo "Deleting genius deployment..."
kubectl delete -f $CWD/jb-gen.yaml
