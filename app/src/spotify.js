var SpotifyWebApi = require('spotify-web-api-node')

/*
 * Calls callback with a json-encoded response of the user's top tracks
 */
function getTopTracks(accessToken, onSuccess, onError) {
  // Instatiate api instance and set its accessToken
  var spotifyApi = new SpotifyWebApi();
  spotifyApi.setAccessToken(accessToken);

  // Make the request and call appropriate callback
  spotifyApi.getMyTopTracks()
  .then(function(data) {
    onSuccess(data);
  }, function(err) {
    onError(err);
  }); 
}


module.exports = {
  getTopTracks : getTopTracks
};

/*
 * Example usage of getTopTracks()
 */
function main() {
  // Set request args
  accessToken =  '';
  onSuccess = function(data) {
    data.body.items.forEach(function(track, index) {
      console.log(track.name + " by " + track.artists[0].name);
    }); 
  }
  onError = function(err) {
    console.log(err);
  }

  getTopTracks(accessToken, onSuccess, onError);
}

if (require.main === module) {
  main();
}

