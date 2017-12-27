sinon = require('sinon');
when = require('when');

app = require('../src/app.js');
spotify = require('../src/spotify.js');
db = require('../src/db.js');

describe('app.js', function() {

  it('should return uri when getPlaylistUri is called', function() {
    var id = 'id';
    var expectedUri = 'uri';
    var expectedObject = {'playlistUri': expectedUri}

    var stub = sinon.stub(db, 'getPlaylistUri');
    stub.resolves(expectedUri);

    return app.getPlaylistUri(id).then(function(actualObject) {
      return sinon.assert.match(actualObject, expectedObject);
    }, catchError);
  });

  it('should return correct Jukebox when generateJukebox is called', function() {
    var accessToken = 'token';
    var userId = 'userId';
    var seedTracks = [{'uri' : 'uri1'}, {'uri' : 'uri2'}];
    var audioFeatures = {'feature1': 0.5, 'feature2': 0.75};

    var playlistId = 1;
    var spotifyPlaylistId = 'playlistId';
    var playlistUri = 'playlistUri';

    var recommendations = [];
    recommendations['body'] = {
      'tracks' : seedTracks
    };

    var playlist = {};
    playlist['body'] = {
      'id' : playlistId,
      'uri' : playlistUri 
    };

    var recommendationsStub = sinon.stub(spotify, 'getRecommendations');
    var createPlaylistStub = sinon.stub(spotify, 'createPlaylist');
    var addTracksStub = sinon.stub(spotify, 'addTracksToPlaylist');
    var dbPlaylistStub = sinon.stub(db, 'addPlaylist');

    recommendationsStub.resolves(recommendations);
    createPlaylistStub.resolves(playlist);
    addTracksStub.resolves(0);
    dbPlaylistStub.resolves(playlistId);

    expectedObject = {
      'playlistUri' : playlistUri,
      'playlistId' : playlistId,
      'accessToken' : accessToken
    };

    return app.generateJukebox(accessToken, userId, seedTracks, audioFeatures).
      then(function(actualObject) {
        sinon.assert.match(actualObject, expectedObject);
      }, catchError);
  });

});

function catchError(err) {
 throw new Error(err); 
}
