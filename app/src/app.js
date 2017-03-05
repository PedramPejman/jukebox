var SpotifyWebApi = require('spotify-web-api-node')

const SEED_TRACKS = 'seed-tracks';
const DEFAULT_PARAMS = 'default-params';

/************** Exported Routines ****************/

/*
 * Returns a promise containing an object of format
 * {SEED_TRACKS: [], DEFAULT_PARAMS: {}}
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
            SEED_TRACKS : seedTracks,
            DEFAULT_PARAMS : defaultParams 
          };
        }, errorHandler);
    }, errorHandler);
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
    'acousticness' : 0,
    'danceability' : 0,
    'energy' : 0,
    'tempo' : 0,
    'valence' : 0
  }
  var numTracks = tracksFeatures.body.audio_features.length;

  // Aggregate every feature for every track
  tracksFeatures.body.audio_features.forEach(function(trackFeatures) {
    Object.keys(features).forEach(function(feature) {
      features[feature] += trackFeatures[feature];
    });   
  });

  // Take simply average of all features
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
  fields = {
    'album': null,
    'artists': [],
    'name': null,
    'id': null,
  };
  
  // Record select fields of each track in seedTracks
  topTracksObj.body.items.forEach(function(track){
    protoTrack = {};
    Object.keys(fields).forEach(function(field) {
      protoTrack[field] = track[field];
    });
    seedTracks.push(protoTrack);
  });

  return seedTracks;
}

/*
 * Generic error handler - should not be used in prod
 */
function errorHandler(err) {
  console.log(err);
}

/******************** Main Routines ********************/

module.exports = {
  getInitialJukeboxState: getInitialJukeboxState
};

/*
 * Example usage of getInitialJukeboxState()
 */
function main() {
  accessToken =  
    'BQD4gqZ79fYkOFjIQwiBPaUtKJLFlaN58lspY5zmtvY9T8RSqMWVq-LMSvTocb4Lvn-cuOkzOyDI9E8zi5s3MR1k90aP4yTfN9UuZTEu6ttWCorX5GM4f7csSNe86ZM6WKwKuPg8C9sS50gGUmsDGQ';
  getInitialJukeboxState(accessToken).then(function(initialState){
    console.log(JSON.stringify(initialState));
  });
}

if (require.main === module) {
  main();
}

