var app = require('./app.js');

var PROTO_PATH = __dirname + '/../protos/app.proto';

var PORT = process.env.JB_APP_SERVICE_PORT || '34000';
var HOST = '0.0.0.0';
var grpc = require('grpc');
var proto = grpc.load(PROTO_PATH).app;

// Jukebox proto fields
const SEED_TRACKS = 'seedTracks';
const AUDIO_FEATURE_PARAMS = 'audioFeatureParams';
const ACCESS_TOKEN = 'accessToken';
const USER_ID = 'userId';
const PLAYLIST_ID = 'playlistId';
const PLAYLIST_URI = 'playlistUri';

/**************** AppService RPC methods  *****************/

/*
 * Stub for the GetInitialJukeboxState RPC method
 */
function getInitialJukeboxState(call, callback) {

  // Unmarshall arguments
  var accessToken = call.request[ACCESS_TOKEN];
  var userId = call.request[USER_ID];

  console.log('Received GetInitialJukeboxState RPC for user ' +  userId + 
      ' with access token : ' + accessToken);

  app.getInitialJukeboxState(accessToken, userId).
    then(function(initialState) {
      console.log(initialState);
      callback(null, initialState);
    }, function(err) {
      console.log(err);
      callback(err, null);
    }); 
}  

/*
 * Stub for the RegisterUser RPC method
 */
function registerUser(call, callback) {
  
  // Unmarshall arguments
  var accessToken = call.request[ACCESS_TOKEN];

  console.log('Received RegisterUser RPC with access token : ' + accessToken);

  app.registerUser(call.request.accessToken).
    then(function(authCreds) {
      console.log(authCreds);
      callback(null, authCreds);
    }, function(err) {
      console.log(err);
      try{
        callback(err, null);
      } catch(e) {
        console.log(e);
        // Because of bug in gRPC implementation cannot marshall some error objects
        callback(1, null);
      }
    }); 
}  

/*
 * Stub for the GenerateJukebox RPC method
 */
function generateJukebox(call, callback) {

  // Unmarshall arguments
  var userId = call.request[USER_ID];
  var accessToken = call.request[ACCESS_TOKEN];
  var seedTracks = call.request[SEED_TRACKS];
  var audioFeatureParams = call.request[AUDIO_FEATURE_PARAMS];

  console.log('Received GenerateJukebox RPC for user ' + userId + 
      'with access token : ' + accessToken + ', seed tracks ' + JSON.stringify(seedTracks) 
      + 'and feature params ' + JSON.stringify(audioFeatureParams));

  app.generateJukebox(accessToken, userId, seedTracks, audioFeatureParams).
    then(function(jukebox) {
      console.log(jukebox);
      callback(null, jukebox);
    }, function(error) {
      console.log(error);
      callback(error, null);
    });
}

/*
 * Stub for the GetPlaylistUri RPC method
 */
function getPlaylistUri(call, callback) {
  
  // Unmarshall arguments
  var id = call.request[PLAYLIST_ID];

  console.log('Received GetPlaylistUri RPC with playlist id : ' + id);

  app.getPlaylistUri(id).
    then(function(uri) {
      console.log(uri);
      callback(null, uri);
    }, function(err) {
      console.log(err);
      callback(err, null);
    }); 
}  

/**************** HandshakeService RPC methods  *****************/

/*
 * Stub for the Shake RPC method for testing purposes 
 */
function shake(call, callback) {
  console.log('Request received with name: ' + call.request.name);
  callback(null, {message: 'Successful handshake with ' + call.request.name}); 
}

/************************ Main methods **************************/


/* 
 * Starts an RPC server that receives requests for the Handshake service at the
 * specified port
 */
function main() {

  // Start up RPC server
  var server = new grpc.Server();

  // Register RPC services
  server.addProtoService(proto.HandshakeService.service, {shake: shake});
  server.addProtoService(proto.AppService.service, {
    getInitialJukeboxState: getInitialJukeboxState,
    registerUser: registerUser,
    generateJukebox: generateJukebox,
    getPlaylistUri: getPlaylistUri
  });

  // Do initialization for app module
  app.init();

  // Bind server to host address and start server 
  server.bind(HOST + ":" + PORT, grpc.ServerCredentials.createInsecure());
  server.start();

  console.log("RPC Server starting at " + HOST + ":" + PORT + "\n");
}

/*
 * If run as a scrip, start the server
 */
if (require.main === module) {
  main();
}
