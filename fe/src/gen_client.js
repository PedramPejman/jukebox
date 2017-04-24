var PROTO_PATH = __dirname + '/../protos/genius.proto';

var HOST = process.env.JB_GEN_SERVICE_HOST || 'localhost';
var PORT = process.env.JB_GEN_SERVICE_PORT || '35000';

var grpc = require('grpc');
var proto = grpc.load(PROTO_PATH).genius;
var genClient = new proto.Genius(HOST + ":" + PORT,
    grpc.credentials.createInsecure());

/*
 * Stub for the GetSeedTracks RPC call
 */
exports.getSeedTracks = function(accessToken, userId, callback) {

  getSeedTracksRequest = {
    authCreds: {
      [ACCESS_TOKEN] : accessToken, 
      [USER_ID] : userId
    }
  };

  genClient.getSeedTracks(getSeedTracksRequest, callback);
}

/*
 * Stub for the Shake RPC call
 */
exports.shake = function(user, callback) {
  genClient.shake({name: user}, callback);
}

/*
 * Example usage of RPC calls
 */
function main() {
  console.log("Contacting " + HOST + ":" + PORT);

  // Set access token and user id for example requests
  var accessToken = '';
  var userId = '';
  
  // Create template callback
  callback = function(err, resp) {
    if (err) console.log("Call did not succeed: " + err);
    else console.log("Call succeeded: " + JSON.stringify(resp));
  };

  /*
   * Example usage of getSeedTracks()
   *
   * exports.getSeedTracks(accessToken, userId, callback);
   */
}

if (require.main === module) main();

