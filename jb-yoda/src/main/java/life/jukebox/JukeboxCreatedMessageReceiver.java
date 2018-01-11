package life.jukebox.yoda;

import com.google.cloud.ServiceOptions;
import com.google.cloud.pubsub.v1.AckReplyConsumer;
import com.google.cloud.pubsub.v1.MessageReceiver;
import com.google.cloud.pubsub.v1.Subscriber;
import com.google.pubsub.v1.PubsubMessage;
import com.google.pubsub.v1.SubscriptionName;

/**
 * This class implements the MessageReceiver class and overrides the receiveMessage 
 * method. Once a message is receieved its ID and data are printed to stdout. An 
 * acknowledgement is sent to the publisher. 
 */
public class JukeboxCreatedMessageReceiver implements MessageReceiver {
  @Override
  public void receiveMessage(PubsubMessage message, AckReplyConsumer consumer) {
    System.out.printf("Message [%s]: '%s'\n",
        message.getMessageId(), message.getData().toStringUtf8());
    consumer.ack();
  }
}
