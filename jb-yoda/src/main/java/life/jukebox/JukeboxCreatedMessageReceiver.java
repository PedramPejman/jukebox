package life.jukebox.yoda;

import com.google.cloud.ServiceOptions;
import com.google.cloud.pubsub.v1.AckReplyConsumer;
import com.google.cloud.pubsub.v1.MessageReceiver;
import com.google.cloud.pubsub.v1.Subscriber;
import com.google.pubsub.v1.PubsubMessage;
import com.google.pubsub.v1.SubscriptionName;

public class JukeboxCreatedMessageReceiver implements MessageReceiver {

  @Override
  public void receiveMessage(PubsubMessage message, AckReplyConsumer consumer) {
    System.out.println("Message Id: " + message.getMessageId());
    System.out.println("Data: :" + message.getData().toStringUtf8());
    consumer.ack();
  }
}
