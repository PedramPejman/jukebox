#!/bin/bash

# Delete ingress
echo "Deleting UI service ingress..."
kubectl delete ingress jenkins --namespace jenkins

# Delete secrets
echo "Deleteing secrets..."
kubectl delete secret tls jenkins docker --namespace jenkins 

# Delete services
echo "Delteing UI and discovery services..."
kubectl delete -f service-jenkins.yaml

# Delete deployments
echo "Deleting jenkins deployments..."
kubectl delete -f jenkins.yaml

# Delete jenkins namespace
echo "Deleting jenkins namespace..."
kubectl delete namespace jenkins

# Delete jenkins home directory disk
echo "Deleting jenkins home directory disk..."
gcloud compute disks delete jenkins-home -q

# Delete jenkins home directory image
echo "Deleting jenkins home image..."
gcloud compute images delete jenkins-home-image -q
