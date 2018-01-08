const PubSub = require(`@google-cloud/pubsub`);

// Import Pub/Sub topic identifiers from environment
const topic = process.env.PUBSUB_TOPIC_JUKEBOX_CREATED

function publishMessage (topicName, message) {

  // Instantiate a PubSub client, topic and publisher
  const pubsub = PubSub();
  const topic = pubsub.topic(topicName);
  const publisher = topic.publisher();

  // Publishe message using publisher client
  const dataBuffer = Buffer.from(message);
  return publisher.publish(dataBuffer)
    .then((results) => {
      const messageId = results[0];

      console.log(`Message ${messageId} published.`);

      return messageId;
    });
}

publishMessage(topic, "first message");
