package life.jukebox.yoda;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.logging.Logger;

public class YodaService extends YodaGrpc.YodaImplBase {
  @Override
  public void shake(HandshakeRequest req, StreamObserver<HandshakeResponse> responseObserver) {
    HandshakeResponse reply = HandshakeResponse.newBuilder().setMessage("Hello " + req.getName()).build();
    responseObserver.onNext(reply);
    responseObserver.onCompleted();
  }
}

