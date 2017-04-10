#!/bin/bash

set -e

echo "Building the Application image..."
docker build -t ${AUTHOR}/${APP_IMAGE_NAME}:$TRAVIS_COMMIT -f ${APP_DIR}/Dockerfile ${APP_DIR}

#echo "Building the Frontend image..."
docker build -t ${AUTHOR}/${FE_IMAGE_NAME}:$TRAVIS_COMMIT -f ${FE_DIR}/Dockerfile ${FE_DIR}

#echo "Building the Genius image..."
docker build -t ${AUTHOR}/${GEN_IMAGE_NAME}:$TRAVIS_COMMIT -f ${GEN_DIR}/Dockerfile ${GEN_DIR}

echo "Creating gcloud service key..."
echo ${GCLOUD_SERVICE_KEY_PRD} | base64 --decode -i > ${HOME}/gcloud-service-key.json

echo "Logging into Gcloud..."
gcloud auth activate-service-account --key-file ${HOME}/gcloud-service-key.json

echo "Setting gcloud configuration..."
gcloud --quiet config set project $PROJECT_NAME
gcloud --quiet config set container/cluster $CLUSTER_NAME_PRD
gcloud --quiet config set compute/zone ${CLOUDSDK_COMPUTE_ZONE}
gcloud --quiet container clusters get-credentials $CLUSTER_NAME_PRD

echo "Logging to docker as $DOCKER_USERNAME..."
docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD

echo "Pushing Docker images to repository.."
docker push ${AUTHOR}/${APP_IMAGE_NAME}:$TRAVIS_COMMIT
docker push ${AUTHOR}/${FE_IMAGE_NAME}:$TRAVIS_COMMIT
docker push ${AUTHOR}/${GEN_IMAGE_NAME}:$TRAVIS_COMMIT

kubectl config view
kubectl config current-context

echo "Deploying container images to kubernetes..."
kubectl set image deployment/${APP_DEPLOYMENT_NAME} \
  ${APP_CONTAINER_NAME}=${AUTHOR}/${APP_IMAGE_NAME}:$TRAVIS_COMMIT
kubectl set image deployment/${FE_DEPLOYMENT_NAME} \
  ${FE_CONTAINER_NAME}=${AUTHOR}/${FE_IMAGE_NAME}:$TRAVIS_COMMIT
kubectl set image deployment/${GEN_DEPLOYMENT_NAME} \
  ${GEN_CONTAINER_NAME}=${AUTHOR}/${GEN_IMAGE_NAME}:$TRAVIS_COMMIT

echo "Finished deployment"
