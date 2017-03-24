GEN_PROTOS=./gen/src/

all: build

build: protos

# Copy proto files for app and fe to be dynamically compilation
# Pre-compile genius proto and mark it as a python package directory
protos: .FORCE
	echo fe app | xargs -n 1 cp -rf protos
	python3 -m grpc_tools.protoc -I ./protos --python_out=$(GEN_PROTOS) \
		--grpc_python_out=$(GEN_PROTOS) ./protos/genius.proto 

network:
	sudo ./bash/network.sh

clean:
	rm -rf */protos

.FORCE:
