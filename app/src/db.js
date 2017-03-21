var app = require('./app.js');

var mysql = require('mysql');

var DB_USER = process.env.DB_USER || 'ubuntu';
var DB_PASS = process.env.DB_PASS || 'jukebox';
var DB_HOST = process.env.DB_HOST || '127.0.0.1';
var DB_NAME = process.env.DB_NAME || 'jb_dev';
var DB_POOL = process.env.DB_POOL || 8;

// Create a global reference for the connection pool to the proxy server
var pool = mysql.createConnection({
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
            return reject(error);
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

    // Use a transaction as the two insertions must be atomic 
    pool.beginTransaction(function(err){

      // Reject if we can't begin transaction
      if (err) return reject(err);

      // Execute query for adding new TopTrack and transition promise state
      pool.query('INSERT INTO `top_track` (`user_id`, `track_uri`) VALUES (?, ?)',
          [userId, trackUri], function(error, results, fields) {
            
            // Rollback first insert if unsuccessful
            if (error) {
                return pool.rollback(function() {
                  return reject(error);
              });
            }

            // Execute query for adding new TrackFeatures and transition promise state
            pool.query('INSERT INTO `track_features` (`track_uri`, `acousticness`, `danceability`, `energy`, `tempo`, `valence`) VALUES (?, ?, ?, ?, ?, ?)',
                [trackUri, features[app.ACOUSTICNESS], features[app.DANCEABILITY], features[app.ENERGY], 
                 features[app.TEMPO], features[app.VALENCE]], function(error, results, fields) {
                  
                  // Rollback first insert if unsuccessful
                  if (error) {
                      return pool.rollback(function() {
                        return reject(error);
                    });
                  }

                  // Commit both queries
                  pool.commit(function(err) {
                    if (err) {
                      return pool.rollback(function() {
                        return reject(err);
                      });
                    }

                    return fulfill("Track " + trackUri + " added for user " + userId);
                  });
            });

      });
    });

  });
}
/******************** Helpers *********************/

module.exports = {
  connect: connect,
  disconnect: disconnect,
  addUser: addUser,
  getUser: getUser,
  addTopTrack: addTopTrack
};

/* 
 * Example usage of the exported routines
 */
function main() {

  // Define template callback
  callback = function(message) {
    if (message) console.log(message);
  }

  // Define disconnect callback
  disconnectCB = function(message) {
    callback(message);
    disconnect();
  }

  // Establish connection
  connect().then(callback, callback);

  // Example usage of addUser()
  addUser("user1", "token1").then(callback, callback);

  // Example usage of getUser()
  getUser("token1").then(callback, callback);

  // Example usage of addTopTrack()
  features = {[app.ACOUSTICNESS]: 0.3, [app.DANCEABILITY]: 0.5, 
    [app.ENERGY]: 0.8, [app.TEMPO]: 113.4, [app.VALENCE]: 0.8};
  addTopTrack("user1", "track-rui", features).then(disconnectCB, disconnectCB);
}

/*
 * If run as a script, start the server
 */
if (require.main === module) {
  main();
} 

