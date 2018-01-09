#!/bin/bash

# Compute cwd
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\

# Delete frontend service
echo "Deleting frontend service..."
kubectl delete -f $CWD/service-jb-fe.yaml 

# Delete frontend
echo "Deleting frontend deployment..."
kubectl delete -f $CWD/jb-fe.yaml
