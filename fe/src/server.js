var handler = require('./http_handler.js');

var PORT = process.env.FE_HTTP_PORT || '8080';

var express = require('express'),
    request = require('request'),
    httpServer = express();

// Set Environmental parameters
httpServer.set('views', __dirname + '/views');
httpServer.set('view engine', 'jade');
httpServer.use(express.static('static/stylesheets'));

// Bind to the specified port
httpServer.listen(PORT, function () {
  console.log('Jukebox is running on port ' + PORT + '\n');
});

/**************** Endpoint Routes ****************/

/*  
 * Handles HTTP.GET traffic on '/'
 * Redirects user to Spotify authroization endpoint
 */
httpServer.get('/', handler.home);

/*  
 * Handles HTTP.GET traffic on '/auth-callback'
 * Extracts access_token, executes getInitialJukeboxState RPC
 * and returns response to user.
 */
httpServer.get('/auth-callback', handler.authCallback);

/*  
 * Handles HTTP GET traffic on '/test-rpc'
 * Tests the RPC connection to App.HandshakeService
 */
httpServer.get('/test-rpc', handler.testRpc);

/*
 * Handles HTTP GET traffic on '/test-view'
 * Returns a test HTML page
 */
httpServer.get('/test-view', handler.testView);

/****************** Helpers **********************/

/*
 * Logs error message and kills process
 */
function fatal(message) {
  console.log('fatal: ' + message);
  process.exit(1);
}
