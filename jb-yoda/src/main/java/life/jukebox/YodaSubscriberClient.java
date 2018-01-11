package life.jukebox.yoda;

import com.google.cloud.ServiceOptions;
import com.google.cloud.pubsub.v1.AckReplyConsumer;
import com.google.cloud.pubsub.v1.MessageReceiver;
import com.google.cloud.pubsub.v1.Subscriber;
import com.google.pubsub.v1.PubsubMessage;
import com.google.pubsub.v1.SubscriptionName;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

public class YodaSubscriberClient{
  private static final String PROJECT_ID = getEnvironmentVariable("GCP_PROJECT_ID");
  private static final String SUBSCRIPTION_ID = getEnvironmentVariable("PUBSUB_SUB_JUKEBOX_CREATED_YODA");

    private static String getEnvironmentVariable (String environmentKey) throws IllegalArgumentException {
      if (System.getenv(environmentKey) == null) { 
        throw new IllegalArgumentException("Environment variable " + environmentKey + " does not exist");
      }
      return System.getenv(environmentKey);
    }
  public static void main(String[] args) {
    SubscriptionName subscriptionName = SubscriptionName.of(PROJECT_ID, SUBSCRIPTION_ID);
    Subscriber subscriber = null;
    try {
      subscriber = Subscriber.newBuilder(subscriptionName, new JukeboxCreatedMessageReceiver()).build();
      subscriber.startAsync().awaitRunning();
      while (true){
      }
    } finally {
      if (subscriber != null) {
        subscriber.stopAsync();
      }
    }
  }
}
