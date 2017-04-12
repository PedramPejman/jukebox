var SpotifyWebApi = require('spotify-web-api-node');

/*
 * Returns a promise containing json-encoded list of user's top tracks
 */
exports.getTopTracks = function getTopTracks(accessToken) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Pass optional params 
  var options = {'limit': '20', 'time_range': 'short_term'};

  // Return top tracks wrapped in a promise
  return spotifyApi.getMyTopTracks(options);
}

/*
 * Returns a promise containing json-encoded set of features for 
 * every track in trackIds
 */
exports.getAudioFeatures = function getAudioFeatures(accessToken, trackIds) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Return audio features wrapped in a promise
  return spotifyApi.getAudioFeaturesForTracks(trackIds);
}

/*
 * Returns a promise containing json-encoded User object
 */
exports.getMe = function getMe(accessToken) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Return audio features wrapped in a promise
  return spotifyApi.getMe();
}

/*
 * Returns a promise containing json-encoded Recommendations object
 */
exports.getRecommendations = function getRecommendations(accessToken,
    seedTracks, audioFeatureParams) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Build options
  var options = {};
  options['seed_tracks'] = seedTracks;
  if (audioFeatureParams != undefined) {
    Object.keys(audioFeatureParams).forEach(function(key, index, _array) {
      this['target_' + key] = audioFeatureParams[key];
    }, options);
  }

  // Return recommendations wrapped in a promise
  return spotifyApi.getRecommendations(options);
}

/*
 * Returns a promise containing json-encoded Playlist object
 */
exports.createPlaylist = function createPlaylist(accessToken, userId, name) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Use Jukebox as name if no name supplied
  if (name === undefined || name == '') name = 'Jukebox';

  // Create playlist and return object wrapped in a promise
  return spotifyApi.createPlaylist(userId, name);
}

/*
 * Returns a promise containing json-encoded Snapshot object
 */
exports.addTracksToPlaylist = function addTracksToPlaylist(accessToken, userId, 
    playlistId, tracks) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Create playlist and return object wrapped in a promise
  return spotifyApi.addTracksToPlaylist(userId, 
      playlistId, tracks);
}


