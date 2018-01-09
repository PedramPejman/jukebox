package life.jukebox.yoda;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Server that manages startup/shutdown of a YodaGrpc server.
 */
public class YodaGrpcServer {
  private static final int DEFAULT_JB_YODA_SERVICE_PORT = 37000;
  private static final String DEFAULT_JB_YODA_SERVICE_PORT_ENV_KEY = "JB_YODA_SERVICE_PORT";

  private static final Logger logger = Logger.getLogger(YodaGrpcServer.class.getName());

  private Server server;

  private void start() throws IOException {
    int port = getPort();
    server = ServerBuilder.forPort(port)
      .addService(new YodaService())
      .build()
      .start();

    logger.info("Server started, listening on " + port);
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        System.err.println("*** shutting down gRPC server since JVM is shutting down");
        YodaGrpcServer.this.stop();
        System.err.println("*** server shut down");
      }
    });
  }

  private void stop() {
    if (server != null) {
      server.shutdown();
    }
  }

  /**
   * Await termination on the main thread since the grpc library uses daemon threads.
   */
  private void blockUntilShutdown() throws InterruptedException {
    if (server != null) {
      server.awaitTermination();
    }
  }

  private static int getPort(){
    return System.getenv("JB_YODA_SERVICE_PORT") != null ? 
      Integer.parseInt(System.getenv("JB_YODA_SERVICE_PORT")) : DEFAULT_JB_YODA_SERVICE_PORT;
  }

  /**
   * Main launches the server from the command line.
   */
  public static void main(String[] args) throws IOException, InterruptedException {
    final YodaGrpcServer server = new YodaGrpcServer();
    server.start();
    server.blockUntilShutdown();
  }

}
