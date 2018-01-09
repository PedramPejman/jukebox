var mysql = require('mysql');

var DB_USER = process.env.DB_USER || 'ubuntu';
var DB_PASS = process.env.DB_PASS || 'jukepass';
var DB_HOST = process.env.DB_HOST || '127.0.0.1';
var DB_NAME = process.env.DB_NAME || 'jb_dev';
var DB_POOL = process.env.DB_POOL || 8;

// Audio Features
const ACOUSTICNESS = 'acousticness';
const DANCEABILITY = 'danceability';
const ENERGY = 'energy';
const TEMPO = 'tempo';
const VALENCE = 'valence';

// Create a global reference for the connection pool to the proxy server
var pool = mysql.createPool({
  connection_limit: DB_POOL,      // 32 in prod
  host: DB_HOST,                  // cloudsql-proxy in prod
  user: DB_USER,                  // From Kubernetes secrets in prod
  password: DB_PASS,              // Null in prod
  database: DB_NAME               // jb_prod in prod
});

/****************** Exported routines *****************/

/*
 * Returns a promise which resolves to fulfilled if
 * the connection to database is successful.
 */
function connect() {
  return new Promise(function (fulfill, reject) {
    // TODO: require all clients to explicitly connect
    return fulfill();
  });
}

/*
 * Returns a promise which resolves to fulfilled if
 * the disconnect from database is successful.
 */
function disconnect() {
  return new Promise(function (fulfill, reject) {

    // Ensure connection pool exists
    if (!pool) return reject("Not connected to DB.");
    
    // Make disconnect call
    pool.end(function(err) {
      if (err) {
        console.log("Could not end database connection pool: " + err);
        return reject(err);
      }
      else {
        console.log("Successfully ended database connection");
        return fulfill();
      }
    });
  });
}

/*
 * Returns a promise which resolves to fulfilled if
 * the insertion is successful.
 */
function addUser(userId, accessToken) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Check if user exists
    pool.query('SELECT * FROM `user` WHERE `id` = ?',
        [userId], function(error, results, fields) {
          if (results && results.length > 0) {

            // Execute query for updating existing user and transition promise state
            pool.query('UPDATE `user` SET `access_token`=? WHERE `id` = ?',
                [accessToken, userId], function(error, results, fields) {
                  if (error) {
                    return reject(error);
                  }
                  return fulfill("User " + userId + " updated");
            });
          } else { 
            
            // Execute query for adding new user and transition promise state
            pool.query('INSERT INTO `user` (`id`, `access_token`) VALUES (?, ?)',
                [userId, accessToken], function(error, results, fields) {
                  if (error) {
                    return reject(error);
                  }
                  return fulfill("User " + userId + " added");
            });
          }
    });
  });
}

/*
 * Returns a promise which resolves to fulfilled and returns the
 * user id if access token corresponds to a valid user.
 */
function getUser(accessToken) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Execute the query and transition promise state
    pool.query('SELECT `id` FROM `user` WHERE `access_token` = ?', 
        [accessToken], function(error, results, fields) {
          if (results === undefined || results.length == 0) {
            return reject(error ? error : 
                "No results found for access_token " + accessToken);
          }
          return fulfill(results[0]['id']);
        });
  });
}

/*
 * Adds the given track to the user's top tracks and 
 * records the track's audio feature parameters.
 * Returns a promise which resolves to fulfilled if
 * the insertion is successful.
 */
function addTopTrack(userId, trackUri, features) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");
    pool.getConnection(function(err, connection){
      
      if (!connection) return reject(err);

      // Use a transaction as the two insertions must be atomic 
      connection.beginTransaction(function(err){

        // Reject if we can't begin transaction
        if (err) {
          connection.release();
          return reject(err);
        }

        // Execute query for adding new TopTrack and transition promise state
        connection.query('INSERT INTO `top_track` (`user_id`, `track_uri`) VALUES (?, ?)',
            [userId, trackUri], function(error, results, fields) {

          // Rollback first insert if unsuccessful
          if (error) {
              return connection.rollback(function() {
                connection.release();
                return reject(error);
            });
          }

          // Execute query for adding new TrackFeatures and transition promise state
          connection.query('INSERT INTO `track_features` (`track_uri`, `acousticness`, \
                `danceability`, `energy`, `tempo`, `valence`) VALUES (?, ?, ?, ?, ?, ?)',
              [trackUri, features[ACOUSTICNESS], features[DANCEABILITY], features[ENERGY], 
              features[TEMPO], features[VALENCE]], function(error, results, fields) {

            // Rollback first insert if unsuccessful
            if (error) {
                return connection.rollback(function() {
                  connection.release();
                  return reject(error);
              });
            }

            // Commit both queries
            connection.commit(function(err) {
              connection.release();
              if (err) {
                return connection.rollback(function() {
                  return reject(err);
                });
              }

              return fulfill();
            });
          });
        });
      });
    });
  });
}


/*
 * Returns a promise which resolves to fulfilled if
 * the insertion is successful.
 */
function addPlaylist(userId, uri) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Execute query for adding a new playlist and transition promise state
    pool.query('INSERT INTO `playlist` (`userId`, `uri`) VALUES (?, ?)',
        [userId, uri], function(error, results, fields) {
          if (error) {
            return reject(error);
          }

          // Query for the auto increment id
          pool.query('SELECT `AUTO_INCREMENT` FROM INFORMATION_SCHEMA.TABLES \
              WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = "playlist";', 
              function(error, results, fields) {
                if (error || results == undefined || results.length < 1) {
                  return reject(error);
                }
                id = results[0]['AUTO_INCREMENT'] - 1;
                return fulfill(id);
              });
    });
  });
}

/*
 * Returns a promise which resolves to fulfilled and returns the
 * playlist id if the uri corresponds to a valid playlist.
 */
function getPlaylistId(uri) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Execute the query and transition promise state
    pool.query('SELECT `id` FROM `playlist` WHERE `uri` = ?', 
        [uri], function(error, results, fields) {
          if (results === undefined || results.length == 0) {
            return reject(error ? error : 
                "No results found for playlist with uri " + uri);
          }
          return fulfill(results[0]['id']);
        });
  });
}

/*
 * Returns a promise which resolves to fulfilled and returns the
 * playlist uri if the id corresponds to a valid playlist.
 */
function getPlaylistUri(id) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Execute the query and transition promise state
    pool.query('SELECT `uri` FROM `playlist` WHERE `id` = ?', 
        [id], function(error, results, fields) {

          if (results === undefined || results.length == 0) {
            return reject(error ? error : 
                "No results found for playlist with id " + id);
          }

          return fulfill(results[0]['uri']);
        });
  });
}

/*
 * Returns a promise which resolves to fulfilled and returns the
 * list of playlist id's and uri's for the given user.
 */
function getUserPlaylists(userId) {
  return new Promise(function (fulfill, reject) {
    
    // Ensure connection exists
    if (!pool) return reject("Not connected to DB.");

    // Execute the query and transition promise state
    pool.query('SELECT `id`, `uri` FROM `playlist` WHERE `userId` = ?', 
        [userId], function(error, results, fields) {

          if (results === undefined || results.length == 0) {
            return reject(error ? error : 
                "No results found for user " + userId);
          }

          // Purify the playlists object
          playlists = results.map(function(item) {
            return {'id': item['id'], 'uri': item['uri']};
          });

          return fulfill(playlists);
        });
  });
}
/******************** Helpers *********************/

module.exports = {
  connect: connect,
  disconnect: disconnect,
  addUser: addUser,
  getUser: getUser,
  addTopTrack: addTopTrack, 
  addPlaylist: addPlaylist,
  getPlaylistId: getPlaylistId,
  getPlaylistUri: getPlaylistUri,
  getUserPlaylists: getUserPlaylists
};

/* 
 * Example usage of the exported routines
 */
function main() {

  /*
   * Example template callback
   *
   * callback = function(message) {
   *  if (message) console.log(message);
   * }
   *
   */

  /*
   * Example usage of connect and disconnect
   *
   * connect().then(callback, callback);
   * disconnect().then(callback, callback);
   */

  /*
   * Example usage of User routines
   *
   * addUser("user1", "token1").then(callback, callback);
   * getUser("token1").then(callback, callback);
   */

  /*
   * Example usage of TopTrack routines
   *
   * features = {[ACOUSTICNESS]: 0.3, [DANCEABILITY]: 0.5, 
   *   [ENERGY]: 0.8, [TEMPO]: 113.4, [VALENCE]: 0.8};
   * addTopTrack("user1", "uri1", features).then(callback, callback);
   */

  /*
   * Example usage of Playlist routines
   *
   * addPlaylist("user1", "uri2").then(disconnectCB, disconnectCB);
   * getPlaylistId("uri1").then(disconnectCB, disconnectCB);
   * getUserPlaylists("user1").then(disconnectCB, disconnectCB);
   * getPlaylistUri("15").then(disconnectCB, disconnectCB);
   */

}

/*
 * If run as a script, start the server
 */
if (require.main === module) {
  main();
} 

