var appClient = require('./app_client.js');

var PORT = process.env.FE_HTTP_PORT || '8080';
var CLIENT_ID = process.env.CLIENT_ID || 'c99f31ef396d40ffb498f24d1803b17f';
//var FE_HTTP_HOST = process.env.FE_HTTP_HOST || 'http://104.197.80.118';
var FE_HTTP_HOST = process.env.FE_HTTP_HOST || 'http://plato.cs.virginia.edu/~pp5nv';
var CLIENT_SECRET = process.env.CLIENT_SECRET;

var express = require('express');
var request = require('request');
var httpServer = express();

// TODO: Add check for all required env vars

/*  
 * Handles HTTP.GET traffic on '/'
 * Redirects user to Spotify authroization endpoint
 */
httpServer.get('/', function (req, res) {
  
  // Set scopes, response type, and redirect uri
  var scopes = 'user-top-read';
  var responseType = 'code';
  var redirectUri = FE_HTTP_HOST + '/auth-callback';

  res.redirect('https://accounts.spotify.com/authorize' + 
      '?response_type=' + responseType +
      '&client_id=' + CLIENT_ID +
      '&scope=' + encodeURIComponent(scopes) +
      '&redirect_uri=' + encodeURIComponent(redirectUri));
});

/*  
 * Handles HTTP.GET traffic on '/auth-callback'
 * Extracts access_token, executes getInitialJukeboxState RPC
 * and returns response to user.
 */
httpServer.get('/auth-callback', function (req, res) {
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
    request(options, function(error, response, body) {
      if (!error && response.statusCode == 200) {

        // Retrieve access token
        accessToken = body.access_token; 

        // Make RPC to obtain initial Jukebox state
        // TODO: Make more readable by using templated error routines
        appClient.getInitialJukeboxState(accessToken, function(err, resp) {
          if (err) { 
            res.send(err);
            console.log(err);
          }
          else {
            res.send(resp);
            console.log(resp);
          }
        });
      } else {
        if (error) {
          console.log(error);
          res.end("Error: " + error);
        } else {
          console.log(response);
          res.end("Request did not receive 200");
        }
      }
    });


  } else {
    console.log("code was not supplied.");
    // TODO: Replace with error page redirect
    res.send("code was not supplied."); 
  }
});

/*  
 * Handles HTTP GET traffic on '/test'
 * Tests the RPC connection to App.HandshakeService
 */
httpServer.get('/test', function (req, res) {
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
});

/*
 * Binds to the specified port
 */
httpServer.listen(PORT, function () {
  console.log('Jukebox is running on port ' + PORT + '\n');
});

