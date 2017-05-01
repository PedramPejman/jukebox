var appClient = require('./app_client.js'),
    request = require('request');

var CLIENT_ID = process.env.CLIENT_ID || 'c99f31ef396d40ffb498f24d1803b17f',
    FE_HTTP_HOST = process.env.FE_HTTP_HOST || 'http://jukebox.life',
    CLIENT_SECRET = process.env.CLIENT_SECRET || fatal("No CLIENT_SECRET"),
    ENVIRONMENT = process.env.ENVIRONMENT || 'development';

// HTTP.GET parameters
const PARAM_ACCESS_TOKEN = 'access_token';
const PARAM_USER_ID = 'user_id';

// JukeboxState member fields
const SEED_TRACKS = 'seedTracks';
const AUDIO_FEATURE_PARAMS = 'audioFeatureParams';
const ACCESS_TOKEN = 'accessToken';
const USER_ID = 'userId';

// Jukebox member fields
const PLAYLIST_URI = 'playlistUri';
const PLAYLIST_ID = 'playlistId';

// Audio Features
const AUDIO_FEATURE_ACOUSTICNESS = 'acousticness';
const AUDIO_FEATURE_DANCEABILITY = 'danceability';
const AUDIO_FEATURE_ENERGY = 'energy';
const AUDIO_FEATURE_TEMPO = 'tempo';
const AUDIO_FEATURE_VALENCE = 'valence';

// Audiofeature Parameter fields
// TODO: obtain from proto definitions 
const AUDIO_FEATURE_FIELDS = [
  AUDIO_FEATURE_ACOUSTICNESS,
  AUDIO_FEATURE_DANCEABILITY,
  AUDIO_FEATURE_ENERGY,
  AUDIO_FEATURE_TEMPO,
  AUDIO_FEATURE_VALENCE];

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
      '&state=' + ENVIRONMENT  +
      '&redirect_uri=' + encodeURIComponent(redirectUri));
};

/*  
 * Extracts access_token, executes getInitialJukeboxState RPC
 * and returns response to user.
 */
exports.authCallback = function (req, res) {

  // If in development, send back to localhost
  if (req.query.state == 'development') {
    res.redirect('http://localhost:8080/auth-callback' +
        '?code='+req.query.code);
    return;
  }

  // Check if authorization code was supplied by Spotify API
  if (req.query.code) {
    // Create GET request options
    var options = buildOptionsForAccessTokenRequest(req.query.code);

    // Call Spotify API to obtain access token and redirect to init
    request(options, function(err, resp, body) {
      registerUserAndRedirectToInit(res, err, resp, body);
    });

  // For development, allow direct receipt of access token
  } else if (ENVIRONMENT != 'production' && req.query.access_token) {
    registerUserAndRedirectToInit(res, null, res, req.query);
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

/*
 * Executes APP.GetInitialJukeboxState RPC and returns
 * resulting JukeboxState to user
 */
exports.initialize = function(req, res) {
  if (!req.query[PARAM_ACCESS_TOKEN] || !req.query[PARAM_USER_ID]) {
    // TODO: Replace with error page
    return echoBack('Required parameters not supplied', null, res);
  }

  var accessToken = req.query[PARAM_ACCESS_TOKEN];
  var userId = req.query[PARAM_USER_ID];

  // Make RPC to obtain initial Jukebox state
  appClient.getInitialJukeboxState(accessToken, userId, function(err, resp) {
    if (err) {
      return echoBack(err, null, res);
    }

    var seedTracks = resp[SEED_TRACKS];
    var features = resp[AUDIO_FEATURE_PARAMS];

    // Prepare audio feature params for presentation
    Object.keys(features).forEach(function(feature) {
      features[feature] = transformAudioFeatureForUI(feature, features[feature]);
    });

    res.render('initialize', {
      [SEED_TRACKS] : seedTracks,
      [AUDIO_FEATURE_PARAMS]: features,
      [ACCESS_TOKEN]: accessToken,
      [USER_ID]: userId});
  });
}

/*
 * Executes APP.GenerateJukebox RPC and redirects
 * the user to /play with the resulting jukebox id.
 */
exports.generate = function(req, res) {
  if (!req.body[USER_ID] || !req.body[ACCESS_TOKEN])
    return echoBack("Request parameters not provided", null, res);

  var userId = req.body[USER_ID];
  var accessToken = req.body[ACCESS_TOKEN];
  var audioFeatures = {};
  var seedTracks = [];

  // Clean parameters for RPC
  Object.keys(req.body).forEach(function(param) {
    if (AUDIO_FEATURE_FIELDS.indexOf(param) > -1)
      audioFeatures[param] = transformAudioFeatureForBackend(param, req.body[param]);
    else if (param != USER_ID && param != ACCESS_TOKEN)
      // TODO: Encapsulate in separate Request/Response in protos
      seedTracks.push({'uri': param.split(":")[2]});
  });

  // Make sure we have appropriate request parameters
  // TODO: Replace with error page
  if (seedTracks.length == 0) 
    return echoBack("No seed tracks selected", null, res);
  if (audioFeatures == {})
    return echoBack("No audio feature parameters were supplied", null, res);

  // Make GenerateJukebox RPC call
  appClient.generateJukebox(accessToken, userId, 
      audioFeatures, seedTracks, function(err, resp) {
        if (err) {
          return echoBack(err, null, res);
        }

        // Redirect user to /play 
        var playlistId = resp[PLAYLIST_ID];
        res.redirect('/play?' + PLAYLIST_ID + '=' + playlistId);
      });
};

/*
 * Executes APP.GetPlaylistUri RPC and shows user the
 * corresponding Jukebox playlist widget.
 */
exports.play = function(req, res) {
  if(!req.query[PLAYLIST_ID])
    return echoBack("Playlist id was not provided", null, res);

  var playlistId = parseInt(req.query[PLAYLIST_ID]);

  // Make GetPlaylistUri RPC call
  appClient.getPlaylistUri(playlistId, function(err, resp) {
    if (err) {
      return echoBack(err, null, res);
    }

    // Show the playlist widget
    uri = encodeURIComponent(resp[PLAYLIST_URI]);
    res.render('play', {'uri': uri});
  });
}

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
 * Creates the options for access token GET request
 */
function buildOptionsForAccessTokenRequest(code) {
  var grantType = 'authorization_code';
  var redirectUri = FE_HTTP_HOST + '/auth-callback';
  var authUrl = 'https://accounts.spotify.com/api/token';
  
  return {
    url : authUrl,
    method : "POST",
    json : true,
    form : {
      grant_type : grantType,
      code : code,
      redirect_uri : redirectUri,
      client_id : CLIENT_ID,
      client_secret : CLIENT_SECRET
    }
  };
}

/*
 * Called in auth-callback 
 * Extracts access token, makes RegisterUser RPC and
 * Redirects user to /initialize
 */
function registerUserAndRedirectToInit(res, error, response, body) {
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
      userId = authCreds[USER_ID];
      res.redirect('/initialize?' + PARAM_USER_ID + '=' + userId 
          + '&' + PARAM_ACCESS_TOKEN + '=' + accessToken);
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

/*
 * Transforms the scale of an audio feature
 * to an easy, human-readable value. Tempo is rounded and
 * all other features are transformed from [0,1) to [0,100)
 */
function transformAudioFeatureForUI(feature, featureValue) {
  if (feature == AUDIO_FEATURE_TEMPO) 
    return Math.round(featureValue);
  return Math.round(featureValue * 100);
}

/*
 * Undoes the transformation in transformAudioFeatureForUI()
 * to its real scale for backend use.
 */
function transformAudioFeatureForBackend(feature, featureValue) {
  featureValue = parseInt(featureValue);
  if (feature == AUDIO_FEATURE_TEMPO)
    return featureValue;
  return (featureValue / 100.0);
}
