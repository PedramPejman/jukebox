var appClient = require('./app_client.js'),
    request = require('request');

var CLIENT_ID = process.env.CLIENT_ID || 'c99f31ef396d40ffb498f24d1803b17f',
    FE_HTTP_HOST = process.env.FE_HTTP_HOST || 'http://jukebox.life',
    CLIENT_SECRET = process.env.CLIENT_SECRET || fatal("No CLIENT_SECRET");
/*  
 * Redirects user to Spotify authroization endpoint
 */
exports.home = function (req, res) {
  
  // Set scopes, response type, and redirect uri
  var scopes = 'user-top-read';
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
      if (!error && response.statusCode == 200) {

        // Retrieve access token
        accessToken = body.access_token; 

        // Make RPC to obtain initial Jukebox state
        // TODO: Make more readable by using templated error routines
        appClient.getInitialJukeboxState(accessToken, function(err, resp) {
          if (err) { 
            res.send(err);
            console.log(err);
          } else {
            res.send(resp);
            console.log(resp);
          }
        });
      } else {
        // Echo error back appropriately
        if (error) {
          console.log(error);
          res.end("Error: " + error);
        } else {
          console.log(response);
          res.end("Request did not receive 200");
        }
      }
    }
