var handler = require('./http_handler.js');

var PORT = process.env.FE_HTTP_PORT || '8080';
var IS_PROD = process.env.ENVIRONMENT == 'production';

var express = require('express'),
    request = require('request'),
    sassMiddleware = require('node-sass-middleware'),
    bodyParser = require('body-parser'),
    httpServer = express();

var APP_ROOT = __dirname;
var STATIC_ROOT = APP_ROOT + '/static';
var VIEWS_ROOT = APP_ROOT + '/views';
var STYLESHEETS_ROOT = STATIC_ROOT + '/stylesheets';
var JS_ROOT = STATIC_ROOT + '/js';
var SASS_ROOT = STATIC_ROOT + '/sass';

var TEMPLATE_ENGINE = 'jade';

/****************** Server Configuration *******************/

// Set views root path
httpServer.set('views', VIEWS_ROOT);

// Set template engine to jade
httpServer.set('view engine', TEMPLATE_ENGINE);

// Set up Sass middleware
httpServer.use(sassMiddleware({
    src: SASS_ROOT,
    dest: STYLESHEETS_ROOT,
    debug: !IS_PROD,
    outputStyle: IS_PROD ? 'compressed' : 'extended'
}));

// Declare directories to look in for static files
httpServer.use(express.static(STYLESHEETS_ROOT));
httpServer.use(express.static(JS_ROOT));

// Configure body parser for POST requests
httpServer.use(bodyParser.json());
httpServer.use(bodyParser.urlencoded({ extended: true }));

// Bind to the specified port
httpServer.listen(PORT, function () {
  console.log('Jukebox is running on port ' + PORT + '\n');
});

/******************** Endpoint Routes ********************/

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

/*
 * Handles HTTP GET traffic on '/home-view'
 * Returns application home page
 */
httpServer.get('/home-view', handler.homeView);

/*
 * Handles HTTP GET traffic on '/initialize'
 * Returns application home page
 */
httpServer.get('/initialize', handler.initialize);

/*
 * Handles HTTP POST traffic on '/generate'
 * Returns application home page
 */
httpServer.post('/generate', handler.generate);

/*
 * Handles HTTP GET traffic on '/play'
 * Returns Spotify playlist widget
 */
httpServer.get('/play', handler.play);

/********************** Helpers **************************/

/*
 * Logs error message and kills process
 */
function fatal(message) {
  console.log('fatal: ' + message);
  process.exit(1);
}
