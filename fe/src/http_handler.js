var appClient = require('./app_client.js'),
    request = require('request');

var CLIENT_ID = process.env.CLIENT_ID || 'c99f31ef396d40ffb498f24d1803b17f',
    FE_HTTP_HOST = process.env.FE_HTTP_HOST || 'http://jukebox.life',
    CLIENT_SECRET = process.env.CLIENT_SECRET || fatal("No CLIENT_SECRET"),
    IS_PROD = process.env.ENVIRONMENT == 'production';

// JukeboxState member fields
const SEED_TRACKS = 'seedTracks';
const AUDIO_FEATURE_PARAMS = 'audioFeatureParams';
const ACCESS_TOKEN = 'accessToken';
const USER_ID = 'userId';

/*  
 * Redirects user to Spotify authroization endpoint
 */
exports.home = function (req, res) {
  
  // Set scopes, response type, and redirect uri
  var scopes = 'user-top-read playlist-modify-public playlist-modify-public';
  var responseType = 'code';
  var redirectUri = FE_HTTP_HOST + '/auth-callback';

  res.redirect('https://accounts.spotify.com/authorize' + 
      '?response_type=' + responseType +
      '&client_id=' + CLIENT_ID +
      '&scope=' + encodeURIComponent(scopes) +
      '&redirect_uri=' + encodeURIComponent(redirectUri));
};

/*  
 * Extracts access_token, executes getInitialJukeboxState RPC
 * and returns response to user.
 */
exports.authCallback = function (req, res) {
  // Check if authorization code was supplied by Spotify API
  if (req.query.code) {
    var code = req.query.code;
    var grantType = 'authorization_code';
    var redirectUri = FE_HTTP_HOST + '/auth-callback';
    var authUrl = 'https://accounts.spotify.com/api/token';
    
    var options = {
      url : authUrl,
      method : "POST",
      json : true,
      form : {
        grant_type : grantType,
        code : code,
        redirect_uri : encodeURIComponent(redirectUri),
        client_id : CLIENT_ID,
        client_secret : CLIENT_SECRET
      }
    };

    // Call Spotify API to obtain access token
    request(options, function(err, resp, body) {
      // Extract access code and make getInitJBState RPC
      getAccessCodeCallback(res, err, resp, body);
    });

  } else if (!IS_PROD && req.query.access_token) {
    // For development, allow direct receipt of access token
    getAccessCodeCallback(res, null, res, req.query);
  } else {
    console.log("code was not supplied.");
    // TODO: Replace with error page redirect
    res.send("code was not supplied."); 
  }
};

/*  
 * Tests the RPC connection to App.HandshakeService
 */
exports.testRpc = function (req, res) {
  // Send an Handshake RPC to app service
  appClient.shake(
      req.query.name ? req.query.name : 'Pedram', function (err, rpcResp) {
    if (err) { 
      console.log("Error sending RPC");
      res.send("Error sending RPC\n");
    }
    else {
      console.log("Received message from app: " + rpcResp.message); 
      res.send(rpcResp.message + "\n");
    }
  });
};

/*
 * Returns a test HTML page
 */
exports.testView = function(req, res) {
  console.log("Serving test page");
  res.render('testView');
};

/*
 * Returns application home page
 */
exports.homeView = function(req, res) {
  console.log("Serving home page");
  res.render('homeView');
};

/********************** Helpers *************************/

/*
 * Logs error message and kills process
 */
function fatal(message) {
  console.log('fatal: ' + message);
  process.exit(1);
}

/*
 * Called in auth-callback 
 * Extracts access token and makes getInitialJikeboxState RPC
 */
function getAccessCodeCallback(res, error, response, body) {
  if (error || response.statusCode != 200) {
    echoBack(error, response, res);
  } else {

    // Retrieve access token
    accessToken = body.access_token; 

    // Make RPC to register user
    appClient.registerUser(accessToken, function(err, authCreds) {
			if (err) {
				return echoBack(err, null, res);
			}	
      // Make RPC to obtain initial Jukebox state
      userId = authCreds[USER_ID];
      appClient.getInitialJukeboxState(accessToken, userId, function(err, resp) {
        echoBack(err, resp, res);
      });
    });

  }
}

/*
 * Template callback - writes err/resp to pipe and logs to console
 */
function echoBack(err, resp, pipe) {
  if (err) { 
    pipe.send(err);
    console.log(err);
  } else {
    pipe.send(resp);
    console.log(resp);
  }
}
