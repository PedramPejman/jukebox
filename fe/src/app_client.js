var PROTO_PATH = __dirname + '/../protos/app.proto';

var HOST = process.env.APP_RPC_HOST || 'localhost';
var PORT = process.env.APP_RPC_PORT || '34000';

var grpc = require('grpc');
var proto = grpc.load(PROTO_PATH).app;
var handshakeClient = new proto.HandshakeService(HOST + ":" + PORT,
    grpc.credentials.createInsecure());
var appClient = new proto.AppService(HOST + ":" + PORT,
    grpc.credentials.createInsecure());

/*
 * Stub for the GetInitialJukeboxState RPC call
 */
function getInitialJukeboxState(accessToken, callback) {
  appClient.getInitialJukeboxState({accessToken: accessToken}, callback);
}

/*
 * Stub for the RegisterUser RPC call
 */
function registerUser(accessToken, callback) {
  appClient.registerUser({accessToken: accessToken}, callback);
}

/*
 * Stub for the Shake RPC call
 */
function shake(user, callback) {
  handshakeClient.shake({name: user}, callback);
}

/*
 * List of exported entities from the app_client module
 */
module.exports = {
  shake: shake,
  getInitialJukeboxState: getInitialJukeboxState,
  registerUser: registerUser
};

/*
 * Example usage of RPC calls
 */
function main() {
  console.log("Contacting " + HOST + ":" + PORT);

  // Set access token for example requests
  access_token = ''; 
  
  // Create template callback
  callback = function(err, resp) {
    if (err) console.log("Call did not succeed: " + err);
    else console.log("Call succeeded: " + JSON.stringify(resp));
  };

  // Example usage of registerUser()
  registerUser(access_token, callback);

  // Example usage of getInitialJukeboxState()
  getInitialJukeboxState(access_token, callback);
}

if (require.main === module) main();

