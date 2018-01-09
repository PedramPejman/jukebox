#!/bin/bash

# This script sets up jenkins deployment, UI service, discovery service
# and an ingress for the UI service.

# Check if environment variables exist
echo "Checking for required environment variables..."
if [[ -z "$DOCKER_USERNAME" || -z "$DOCKER_PASSWORD" || -z "$JENKINS_PASSWORD" ]]; 
then 
  echo "Fatal: Environment variables not set"; 
  exit 1;
fi

# Download jenkins home directory image
echo "Creating jenkins home directory image..."
gcloud compute images create jenkins-home-image --source-uri \
  https://storage.googleapis.com/solutions-public-assets/jenkins-cd/jenkins-home-v3.tar.gz

# Create disk with jenkins home directory image
echo "Creating jenkins home directory disk..."
gcloud compute disks create jenkins-home --image jenkins-home-image

# Create namespace for jenkins
echo "Creating namespace for jenkins..."
kubectl create ns jenkins

# Upload jenkins role/password secrets to kubernetes
echo "Creating secret for jenkins options..."
kubectl create secret generic jenkins --namespace=jenkins \
  --from-literal=options="--argumentsRealm.passwd.jenkins=$JENKINS_PASSWORD \
  --argumentsRealm.roles.jenkins=admin"

# Upload Docker credentials to kubernetes
echo "Creating secret for docker credentials..."
kubectl create secret generic docker --namespace=jenkins \
  --from-literal=username=$DOCKER_USERNAME \
  --from-literal=password=$DOCKER_PASSWORD 

# Create jenkins deployment
echo "Creating jenkins deployment..."
kubectl apply -f jenkins.yaml

# Create jenkins UI and discovery services
echo "Creating jenkins UI and discovery services..."
kubectl apply -f service-jenkins.yaml

# Create temporary ssl certificate
echo "Creating ssl certificate..."
 openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
   -keyout /tmp/tls.key -out /tmp/tls.crt -subj "/CN=jenkins/O=jenkins"

# Upload ssl certificate secret to kubernetes
echo "Creating secret for ssl certrificate..."
kubectl create secret generic tls --from-file=/tmp/tls.crt \
  --from-file=/tmp/tls.key --namespace jenkins

# Create ingress for the jenkins UI service
echo "Creating ingress for jenkins UI service..."
kubectl apply -f ingress-jenkins.yaml
