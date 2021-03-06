var db = require('./db.js');
var spotify = require('./spotify.js');

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

// JukeboxParams Fields
const JB_PARAM_POPULARITY = 'popularity';
const JB_PARAM_NOVELTY = 'novelty';

// Jukebox proto fields
const SEED_TRACKS = 'seedTracks';
const AUDIO_FEATURE_PARAMS = 'audioFeatureParams';
const JUKEBOX_PARAMS = 'jukeboxParams';
const ACCESS_TOKEN = 'accessToken';
const USER_ID = 'userId';
const PLAYLIST_ID = 'playlistId';
const PLAYLIST_URI = 'playlistUri';

const DEFAULT_PLAYLIST_NAME = 'Jukebox';
const ID = 'id';
const MAX_SEED_TRACKS = 5;

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
 * with format {SEED_TRACKS: [], AUDIO_FEATURE_PARAMS: {}}
 */
function getInitialJukeboxState(accessToken, userId) {

  // Return values
  var seedTracks, defaultParams, topTracksObj;

  // Gets seed tracks and raw AudioFeatures object
  var getSeedsAndFeatures = function(_topTracksObj) {
    topTracksObj = _topTracksObj;
    seedTracks = pruneTracks(topTracksObj.body.items);
    tracksIds = extractTracksIds(topTracksObj);
    return spotify.getAudioFeatures(accessToken, tracksIds)
  };

  // Records top track and returns InitJBState
  var recordAndGetInitState = function(topTracksFeatures) {
    return recordTopTracks(accessToken, topTracksFeatures).then(function() {
      
      featureParams = getInitAudioFeatureParams(topTracksFeatures);
      jukeboxParams = getInitJukeboxParams(topTracksObj.body.items);

      return {
        [SEED_TRACKS] : seedTracks,
        [AUDIO_FEATURE_PARAMS] : featureParams,
        [JUKEBOX_PARAMS] : jukeboxParams,
        [ACCESS_TOKEN] : accessToken,
        [USER_ID] : userId
      };
    }, bubbleUpError);
  };

  return spotify.getTopTracks(accessToken)
    .then(getSeedsAndFeatures, bubbleUpError)
    .then(recordAndGetInitState, bubbleUpError);
}

/*
 * Registers the user by adding them to DB.user.
 * Returns a promise containing the complete authCreds object 
 */
function registerUser(accessToken) {
  return spotify.getMe(accessToken).
    then(function(userObject) {
      id = userObject.body[ID];
      return db.addUser(id, accessToken).then(function() {
        return {
          [USER_ID] : id,
          [ACCESS_TOKEN] : accessToken
        };
      }, bubbleUpError);
    }, bubbleUpError);
}

/*
 * Returns a complete Jukebox object 
 */
function generateJukebox(accessToken, userId, 
    seedTracksObj, audioFeatureParams) {

  // Get Seed tracks and cap at MAX_SEED_TRACKS
  var seedTracks = [];
  seedTracksObj.forEach(function(track, index) {
    if (index < MAX_SEED_TRACKS)
      seedTracks.push(track[SEED_TRACK_URI]);
  });

  // Get recommendations
  return spotify.getRecommendations(accessToken, seedTracks, audioFeatureParams).
    then(function(recommendationsObj) {

      // Prune recommendations object 
      var tracks = [];
      recommendationsObj.body.tracks.forEach(function(track, index) {
        tracks.push(track[SEED_TRACK_URI]); 
      });

      // Create playlist for user
      return spotify.createPlaylist(accessToken, userId, DEFAULT_PLAYLIST_NAME).
        then(function(playlist) {

          // Add tracks to playlist
          var spotifyPlaylistId = playlist.body.id;
          return spotify.addTracksToPlaylist(accessToken, userId, spotifyPlaylistId, tracks).
            then(function() {

              // Add playlist to database
              playlistUri = playlist.body.uri;
              return db.addPlaylist(userId, playlistUri).then(function(playlistId) {
                
                // Return Jukebox object
                return {
                  [PLAYLIST_URI] : playlistUri,
                  [PLAYLIST_ID] : playlistId,
                  [ACCESS_TOKEN] : accessToken
                };

            }, bubbleUpError);
          }, bubbleUpError);
        }, bubbleUpError);
  }, bubbleUpError);
}

/*
 * Retrieves the playlist with the supplied id.
 * Returns a promise containing the playlist uri.
 */
function getPlaylistUri(id) {
  if (id == undefined) return new Promise.reject();

  return db.getPlaylistUri(id).
    then(function(uri) {
      return {
        [PLAYLIST_URI]: uri
      };
    }, bubbleUpError);
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
 * Returns default audio feature parameters
 */
function getInitAudioFeatureParams(tracksFeatures) {
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

  // Take simple average of all features to 2 decimal places
  Object.keys(features).forEach(function(feature) {
    features[feature] = Math.round(100 * features[feature] / numTracks) / 100;  
  });

  return features;
}

/*
 * Returns a partial JukeboxParams object
 */
function getInitJukeboxParams(tracks) {
  popularity = 0;
  numTracks = tracks.length;

  tracks.forEach(function(track) {
    popularity += track[JB_PARAM_POPULARITY]
  });

  popularity = Math.round(100 * popularity / numTracks) / 100;

  return {
    [JB_PARAM_POPULARITY] : popularity
  };
}

/*
 * Returns select fields of a list of tracks
 * TODO: Check existence of each layer before setting 
 */
function pruneTracks(tracks) {
  prunedTracks= [];

  // Record select fields of each track in seedTracks
  tracks.forEach(function(track){
    tempTrack = {};
    tempTrack[SEED_TRACK_NAME] = track[SEED_TRACK_NAME];
    tempTrack[SEED_TRACK_URI] = track[SEED_TRACK_URI];
    tempTrack[SEED_TRACK_ALBUM] = track[SEED_TRACK_ALBUM]['name']; 
    tempTrack[SEED_TRACK_ARTIST] = track['artists'][0]['name'];
    tempTrack[SEED_TRACK_IMAGE] = track[SEED_TRACK_ALBUM]['images'][0]['url'];
    prunedTracks.push(tempTrack);
  });

  return prunedTracks;
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
  registerUser: registerUser,
  generateJukebox: generateJukebox,
  getPlaylistUri: getPlaylistUri
};

/*
 * Example usage of exported routines
 */
function main() {
  
  // Set access token and user id for example calls
  var accessToken = '';

  // Template callback
  var callback = function(message) {
    if (message) console.log(message);
  }
  
  /*
   * Example usage of registerUser 
   *
   * registerUser(accessToken).then(callback, callback);
   */

  /* Example usage of getInitialJukeboxState
   *
   * getInitialJukeboxState(accessToken).then(callback, callback);
   */
}

if (require.main === module) {
  main();
}

