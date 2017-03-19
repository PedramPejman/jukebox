var mysql = require('mysql');

var DB_USER = process.env.DB_USER || 'ubuntu';
var DB_PASS = process.env.DB_PASS || 'jukebox';
var DB_HOST = process.env.DB_HOST || '127.0.0.1';
var DB_NAME = process.env.DB_NAME || 'jb_dev';

// Create a connection to the Mysql proxy server
var con = mysql.createConnection({
  host: DB_HOST,      // cloudsql-proxy in prod
  user: DB_USER,      // From Kubernetes secrets in prod
  password: DB_PASS,  // Null in prod
  database: DB_NAME   // jb_prod in prod
});

/****************** Exported routines *****************/

/*
 * Returns a promise which resolves to fulfilled if
 * the connection to database is successful.
 */
function connect() {
  return new Promise(function (fulfill, reject) {
    con.connect(function(err){
      if(err){
        console.log('Error connecting to database: ' + err);
        reject(err);
      }
      else {
        console.log('Connection to database established');
        fulfill();
      }
    });
  });
}

/*
 * Returns a promise which resolves to fulfilled if
 * the disconnect from database is successful.
 */
function disconnect() {
  return new Promise(function (fulfill, reject) {
    con.end(function(err) {
      if (err) {
        console.log("Could not end database connection: " + err);
        reject(err);
      }
      else {
        console.log("Successfully ended database connection");
        fulfill();
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
    con.query('INSERT INTO `user` (`id`, `access_token`) VALUES (?, ?)',
        [userId, accessToken], function(error, results, fields) {
          if (error) {
            console.log("unable to add user %s with access token %s", userId, accessToken);
            reject(error);
          }
          fulfill();
        });
  });
}

/*
 * Returns a promise which resolves to fulfilled and returns the
 * user id if access token corresponds to a valid user.
 */
function getUser(accessToken) {
  return new Promise(function (fulfill, reject) {
    con.query('SELECT `id` FROM `user` WHERE `access_token` = ?', 
        [accessToken], function(error, results, fields) {
          if (error) {
            console.log("unable to find user with access token %s", accessToken);
            reject(error);
          }
          fulfill(results[0]['id']);
        });
  });
}

/******************** Helpers *********************/

module.exports = {
  connect: connect,
  disconnect: disconnect,
  addUser: addUser
};

/* 
 * Example usage of the exported routines
 */
function main() {
  // Define template callback
  callback = function(message) {
    if (message) console.log(message);
  }

  // Establish connection
  connect();

  // Example usage of addUser()
  addUser("user1", "token1").
    then(callback, callback);

  // Example usage of getUser()
  getUser("token1")
    .then(callback, callback);

  // Terminate connection
  disconnect();
}

/*
 * If run as a script, start the server
 */
if (require.main === module) {
  main();
} 

