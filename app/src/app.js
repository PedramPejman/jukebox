var db = require('./db.js');

var SpotifyWebApi = require('spotify-web-api-node');

/****************** Constants *******************/

// Audio Features
const AUDIO_FEATURE_ACOUSTICNESS = 'acousticness';
const AUDIO_FEATURE_DANCEABILITY = 'danceability';
const AUDIO_FEATURE_ENERGY = 'energy';
const AUDIO_FEATURE_TEMPO = 'tempo';
const AUDIO_FEATURE_VALENCE = 'valence';

// Seed Track Fields
const SEED_TRACK_ALBUM = 'album';
const SEED_TRACK_ARTIST = 'artist';
const SEED_TRACK_NAME = 'name';
const SEED_TRACK_URI = 'uri';
const SEED_TRACK_IMAGE = 'image';

// InitialJukeboxState member fields
const SEED_TRACKS = 'seedTracks';
const DEFAULT_PARAMS = 'defaultParams';
const ACCESS_TOKEN = 'accessToken';

/************** Exported Routines ****************/

/*
 * Initialization routine for all set up work and assertions
 */
function init() {
  // Ensure valid connection pool to database
  db.connect().then(function() {
    console.log("Connected to database");
  }, logError);
}

/*
 * Returns a promise containing the initial jukebox state
 * with format {SEED_TRACKS: [], DEFAULT_PARAMS: {}}
 */
function getInitialJukeboxState(accessToken) {

  // Return values
  var seedTracks, defaultParams;

  // Gets seed tracks and raw AudioFeatures object
  var getSeedsAndFeatures = function(topTracksObj) {
    seedTracks = getSeedTracks(topTracksObj);
    tracksIds = extractTracksIds(topTracksObj);
    return getAudioFeatures(accessToken, tracksIds)
  };

  // Records top track and returns InitJBState
  var recordAndGetInitState = function(topTracksFeatures) {
    return recordTopTracks(accessToken, topTracksFeatures).then(function() {
      defaultParams = getDefaultJukeboxParams(topTracksFeatures);
      return {
        [SEED_TRACKS] : seedTracks,
        [DEFAULT_PARAMS] : defaultParams,
        [ACCESS_TOKEN] : accessToken
      };
    }, bubbleUpError);
  };

  return getTopTracks(accessToken)
    .then(getSeedsAndFeatures, bubbleUpError)
    .then(recordAndGetInitState, bubbleUpError);
}

/*
 * Registers the user by adding them to DB.user.
 * Returns a promise containing a 
 */
function registerUser(accessToken) {
  return getMe(accessToken).
    then(function(userObject) {
      id = userObject['body']['id'];
      return db.addUser(id, accessToken);
    }, bubbleUpError);
}

/************** Spotify API Wrappers ****************/

/*
 * Returns a promise containing json-encoded list of user's top tracks
 */
function getTopTracks(accessToken) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Return top tracks wrapped in a promise
  return spotifyApi.getMyTopTracks();
}

/*
 * Returns a promise containing json-encoded set of features for 
 * every track in trackIds
 */
function getAudioFeatures(accessToken, trackIds) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Return audio features wrapped in a promise
  return spotifyApi.getAudioFeaturesForTracks(trackIds);
}

/*
 * Returns a promise containing json-encoded User object
 */
function getMe(accessToken) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Return audio features wrapped in a promise
  return spotifyApi.getMe();
}

/****************** Helpers ********************/

/*
 * Returns ids of all tracks in a TopTracks Object
 */
function extractTracksIds(topTracksObj) {
  return topTracksObj.body.items.map(function(track) {
    return track.id;
  }); 
}

/*
 * Returns default Jukebox parameters
 */
function getDefaultJukeboxParams(tracksFeatures) {
  features = {
    [AUDIO_FEATURE_ACOUSTICNESS] : 0,
    [AUDIO_FEATURE_DANCEABILITY] : 0,
    [AUDIO_FEATURE_ENERGY] : 0,
    [AUDIO_FEATURE_TEMPO] : 0,
    [AUDIO_FEATURE_VALENCE] : 0
  };

  var numTracks = tracksFeatures.body.audio_features.length;

  // Aggregate every feature for every track
  tracksFeatures.body.audio_features.forEach(function(trackFeatures) {
    Object.keys(features).forEach(function(feature) {
      features[feature] += trackFeatures[feature];
    });   
  });

  // Take simple average of all features
  Object.keys(features).forEach(function(feature) {
    features[feature] = features[feature] / numTracks;  
  });

  return features;
}

/*
 * Returns select fields of potential seed tracks
 */
function getSeedTracks(topTracksObj) {
  seedTracks = [];

  // Record select fields of each track in seedTracks
  topTracksObj.body.items.forEach(function(track){
    tempTrack = {};
    tempTrack[SEED_TRACK_NAME] = track[SEED_TRACK_NAME];
    tempTrack[SEED_TRACK_URI] = track[SEED_TRACK_URI];
    tempTrack[SEED_TRACK_ALBUM] = track[SEED_TRACK_ALBUM]['name']; 
    tempTrack[SEED_TRACK_ARTIST] = track['artists'][0]['name'];
    tempTrack[SEED_TRACK_IMAGE] = track[SEED_TRACK_ALBUM]['images'][0]['url'];
    seedTracks.push(tempTrack);
  });

  return seedTracks;
}

/* 
 * Records the top tracks and their features for the given user
 * Returns 0 on success and non-zero status code on failure
 */
function recordTopTracks(accessToken, topTrackFeatures) {
  template = {
    [AUDIO_FEATURE_ACOUSTICNESS] : 0,
    [AUDIO_FEATURE_DANCEABILITY] : 0,
    [AUDIO_FEATURE_ENERGY] : 0,
    [AUDIO_FEATURE_TEMPO] : 0,
    [AUDIO_FEATURE_VALENCE] : 0
  };
  features = {};

  // Retrieve user id for access token
  return db.getUser(accessToken).then(function(userId) {

    // Prepare data to make db.addTopTrack() call
    topTrackFeatures.body.audio_features.forEach(function(trackFeatures) {

      // Reset features
      features = {};

      // Compile relevant features
      Object.keys(template).forEach(function(feature) {
        features[feature] = trackFeatures[feature];
      });

      // compile track uri
      trackUri = trackFeatures[SEED_TRACK_URI];
      
      // Add top track item
      // TODO: return a promise for each insertion and wait on all
      db.addTopTrack(userId, trackUri, features).catch(function(err) {
        if (err.code != "ER_DUP_ENTRY") logError(err); 
      });
    });

  }, bubbleUpError);
}

/*
 * Bubbles error to parent promise
 */
function bubbleUpError(err) {
  throw new Error(err);
}

/*
 * Logs error to console
 */
function logError(err) {
  console.log(err);
}

/******************** Main Routines ********************/

module.exports = {
  // Constants
  ACOUSTICNESS: AUDIO_FEATURE_ACOUSTICNESS,
  DANCEABILITY: AUDIO_FEATURE_DANCEABILITY,
  ENERGY: AUDIO_FEATURE_ENERGY,
  TEMPO: AUDIO_FEATURE_TEMPO,
  VALENCE: AUDIO_FEATURE_VALENCE,
  
  // Routines
  init: init,
  getInitialJukeboxState: getInitialJukeboxState,
  registerUser: registerUser
};

/*
 * Example usage of exported routines
 */
function main() {
  
  // Set access token for example calls
  accessToken = '';

  // Template callback
  callback = function(message) {
    if (message) console.log(message);
  }
  
  // Example usage of registerUser 
  registerUser(accessToken).then(callback, callback);

  // Example usage of getInitialJukeboxState
  getInitialJukeboxState(accessToken).then(callback, callback);
  
}

if (require.main === module) {
  main();
}

