GEN_PROTOS=./gen/src/protos

all: build

build: protos

protos: .FORCE
	# Copy proto files for app and fe to be dynamically compilation
	echo fe app | xargs -n 1 cp -rf protos
	# Pre-compile genius proto
	python -m grpc_tools.protoc -I ./protos --python_out=$(GEN_PROTOS) --grpc_python_out=$(GEN_PROTOS) ./protos/genius.proto

network:
	sudo ./bash/network.sh

clean:
	rm -rf */protos

.FORCE:
