GEN_PROTOS=./gen/src/

# Builds the FE service locally
build-fe: .FORCE
	cp -rf protos fe

# Builds the APP service locally
build-app: .FORCE
	cp -rf protos app

# Default entrypoint
all: build

# Builds all services
build: protos

# Builds proto files for app and fe
# Compiles genius protos and marks it as a python package directory
protos: .FORCE
	echo fe app | xargs -n 1 cp -rf protos
	python3 -m grpc_tools.protoc -I ./protos --python_out=$(GEN_PROTOS) \
		--grpc_python_out=$(GEN_PROTOS) ./protos/genius.proto 

# Creates a Docker network for local integration
network:
	sudo ./bash/network.sh

# Cleans local files
clean:
	rm -rf */protos */node_modules

# To be used to force execution of a recipe
.FORCE:
