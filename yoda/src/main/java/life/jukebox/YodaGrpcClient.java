package life.jukebox.yoda;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A simple client that requests a greeting from the {@link HelloWorldServer}.
 */
public class YodaGrpcClient {
  private static final Logger logger = Logger.getLogger(YodaGrpcClient.class.getName());

  private final ManagedChannel channel;
  private final YodaGrpc.YodaBlockingStub blockingStub;

  /** Construct client connecting to HelloWorld server at {@code host:port}. */
  public YodaGrpcClient(String host, int port) {
    this(ManagedChannelBuilder.forAddress(host, port)
        // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
        // needing certificates.
        .usePlaintext(true)
        .build());
  }

  /** Construct client for accessing RouteGuide server using the existing channel. */
  YodaGrpcClient(ManagedChannel channel) {
    this.channel = channel;
    blockingStub = YodaGrpc.newBlockingStub(channel);
  }

  public void shutdown() throws InterruptedException {
    channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
  }

  /** Say hello to server. */
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

  /**
   * Yoda server. If provided, the first element of {@code args} is the name to use in the
   * greeting.
   */
  public static void main(String[] args) throws Exception {
    YodaGrpcClient client = new YodaGrpcClient("localhost", 37000);
    try {
      /* Access a service running on the local machine on port 37000 */
      String user = "world";
      if (args.length > 0) {
        user = args[0]; /* Use the arg as the name to greet if provided */
      }
      client.shake(user);
    } finally {
      client.shutdown();
    }
  }
}
