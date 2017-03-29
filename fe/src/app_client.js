var PROTO_PATH = __dirname + '/../protos/app.proto';

var HOST = process.env.APP_RPC_HOST || 'localhost';
var PORT = process.env.APP_RPC_PORT || '34000';

var grpc = require('grpc');
var proto = grpc.load(PROTO_PATH).app;
var handshakeClient = new proto.HandshakeService(HOST + ":" + PORT,
    grpc.credentials.createInsecure());
var appClient = new proto.AppService(HOST + ":" + PORT,
    grpc.credentials.createInsecure());

// Jukebox proto fields
const SEED_TRACKS = 'seedTracks';
const AUDIO_FEATURE_PARAMS = 'audioFeatureParams';
const ACCESS_TOKEN = 'accessToken';
const USER_ID = 'userId';
const PLAYLIST_ID = 'playlistId';
const PLAYLIST_URI = 'playlistUri';

/*
 * Stub for the GetInitialJukeboxState RPC call
 */
function getInitialJukeboxState(accessToken, userId, callback) {
  authCreds = {
    [ACCESS_TOKEN] : accessToken, 
    [USER_ID] : userId
  };

  appClient.getInitialJukeboxState(authCreds, callback);
}

/*
 * Stub for the RegisterUser RPC call
 */
function registerUser(accessToken, callback) {
  authCreds = {
    [ACCESS_TOKEN] : accessToken
  };

  appClient.registerUser(authCreds, callback);
}

/*
 * Stub for the GenerateJukebox RPC call
 */
function generateJukebox(accessToken, userId, 
    audioFeatureParams, seedTracks, callback) {

  jukeboxState = {
    [ACCESS_TOKEN] : accessToken,
    [USER_ID] : userId,
    [AUDIO_FEATURE_PARAMS] : audioFeatureParams,
    [SEED_TRACKS] : seedTracks 
  };

  appClient.generateJukebox(jukeboxState, callback);
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
  registerUser: registerUser,
  generateJukebox: generateJukebox
};

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
   *  Example usage of registerUser()
   *
   * registerUser(accessToken, callback);
   */

  /*
   * Example usage of getInitialJukeboxState()
   *
   * getInitialJukeboxState(accessToken, userId, callback);
   */
  
  /*
   * Example usage of generateJukebox()
   *
   * var audioFeatures = {'energy' : .6};
   * var seedTracks = [{'uri': '6VwBbL8CzPiC4QV66ay7oR'}]; 
   * generateJukebox(accessToken, userId, audioFeatures, seedTracks, callback);
   */
}

if (require.main === module) main();

