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
const SEED_TRACK_ID = 'id';
const SEED_TRACK_IMAGE = 'image';

// InitialJukeboxState member fields
const SEED_TRACKS = 'seedTracks';
const DEFAULT_PARAMS = 'defaultParams';

/************** Exported Routines ****************/

/*
 * Returns a promise containing the initial jukebox state
 * with format {SEED_TRACKS: [], DEFAULT_PARAMS: {}}
 */
function getInitialJukeboxState(accessToken) {
  return getTopTracks(accessToken)
    .then(function(topTracksObj) {
      seedTracks = getSeedTracks(topTracksObj);
      tracksIds = extractTracksIds(topTracksObj);
      return getAudioFeatures(accessToken, tracksIds)
        .then(function(tracksFeatures) {
          defaultParams = getDefaultJukeboxParams(tracksFeatures);
          return {
            [SEED_TRACKS] : seedTracks,
            [DEFAULT_PARAMS] : defaultParams 
          };
        }, bubbleUpError);
    }, getEmptyInitialJukeboxState);
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
 * Returns a promise containing default Jukebox parameters
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
 * Returns a promise containing select fields of potential seed tracks
 */
function getSeedTracks(topTracksObj) {
  seedTracks = [];

  // Record select fields of each track in seedTracks
  topTracksObj.body.items.forEach(function(track){
    tempTrack = {};
    tempTrack[SEED_TRACK_NAME] = track[SEED_TRACK_NAME];
    tempTrack[SEED_TRACK_ID] = track[SEED_TRACK_ID];
    tempTrack[SEED_TRACK_ALBUM] = track[SEED_TRACK_ALBUM]['name']; 
    tempTrack[SEED_TRACK_ARTIST] = track['artists'][0]['name'];
    tempTrack[SEED_TRACK_IMAGE] = track[SEED_TRACK_ALBUM]['images'][0]['url'];
    seedTracks.push(tempTrack);
  });

  return seedTracks;
}

/*
 * Bubbles error to parent promise
 */
function bubbleUpError(err) {
  return err;
}

/*
 * Returns an empty InitialJukeboxState object
 */
function getEmptyInitialJukeboxState(error) {
  // Log why we are returning empty initial state
  console.log(error);

  return {
    [SEED_TRACKS] : null, 
    [DEFAULT_PARAMS] : null
  };
}

/******************** Main Routines ********************/

module.exports = {
  getInitialJukeboxState: getInitialJukeboxState,
};

/*
 * Example usage of getInitialJukeboxState()
 */
function main() {
  accessToken = ''; 

  getInitialJukeboxState(accessToken).
    then(function(initialState){
    console.log(initialState);
  }, function(err){
    console.log(err);
  });
}

if (require.main === module) {
  main();
}

