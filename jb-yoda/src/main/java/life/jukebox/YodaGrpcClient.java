package life.jukebox.yoda;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A simple client that sends a greeting request to the YodaGrpcServer.
 */
public class YodaGrpcClient {
  private static final int DEFAULT_JB_YODA_SERVICE_PORT = 37000;
  private static final String DEFAULT_JB_YODA_SERVICE_HOST = "localhost";
  private static final String JB_YODA_SERVICE_HOST_ENV_KEY = "JB_YODA_SERVICE_HOST";
  private static final String JB_YODA_SERVICE_PORT_ENV_KEY = "JB_YODA_SERVICE_PORT";

  private static final Logger logger = Logger.getLogger(YodaGrpcClient.class.getName());

  private final ManagedChannel channel;
  private final YodaGrpc.YodaBlockingStub blockingStub;

  public YodaGrpcClient(String host, int port) {
    this(ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext(true)
        .build());
  }

  YodaGrpcClient(ManagedChannel channel) {
    this.channel = channel;
    blockingStub = YodaGrpc.newBlockingStub(channel);
  }

  public void shutdown() throws InterruptedException {
    channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
  }

  public void shake(String name) {
    logger.info("Will try to greet " + name + " ...");
    HandshakeRequest request = HandshakeRequest.newBuilder().setName(name).build();
    HandshakeResponse response;
    try {
      response = blockingStub.shake(request);
    } catch (StatusRuntimeException e) {
      logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
      return;
    }
    logger.info("Greeting: " + response.getMessage());
  }

  public static void main(String[] args) throws Exception {
    String host = System.getenv(JB_YODA_SERVICE_HOST_ENV_KEY) != null ?
      System.getenv(JB_YODA_SERVICE_HOST_ENV_KEY) : DEFAULT_JB_YODA_SERVICE_HOST;

    int port = System.getenv(JB_YODA_SERVICE_PORT_ENV_KEY) != null ?
      Integer.parseInt(System.getenv(JB_YODA_SERVICE_PORT_ENV_KEY)) : DEFAULT_JB_YODA_SERVICE_PORT;

    YodaGrpcClient client = new YodaGrpcClient(host, port);
    try {
      String user = "Peyman";
      client.shake(user);
    } finally {
      client.shutdown();
    }
  }
}
