var PROTO_PATH = __dirname + '/../protos/genius.proto';

var HOST = process.env.GEN_RPC_HOST || 'localhost';
var PORT = process.env.GEN_RPC_PORT || '35000';

var grpc = require('grpc');
var proto = grpc.load(PROTO_PATH).genius;
var handshakeClient = new proto.Handshake(HOST + ":" + PORT,
    grpc.credentials.createInsecure());

/*
 * Stub for the Shake RPC call
 */
function shake(user, callback) {
  handshakeClient.shake({name: user}, callback);
}

/*
 * List of exported entities from the gen_client module
 */
module.exports = {
  shake: shake,
};

/*
 * Example usage of RPC calls
 */
function main() {
  console.log("Contacting " + HOST + ":" + PORT);

  // Create template callback
  callback = function(err, resp) {
    if (err) console.log("Call did not succeed: " + err);
    else console.log("Call succeeded: " + JSON.stringify(resp));
  };

  // Example usage of shake()
  shake("Pedram", callback);
}

if (require.main === module) main();

