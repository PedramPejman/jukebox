var mysql = require('mysql');

var DB_USER = process.env.DB_USER || fatal("DB_USER not an environment variable");
var DB_PASS = process.env.DB_PASS || '';
var DB_HOST = process.env.DB_HOST || '127.0.0.1';

// Create a connection to the Mysql proxy server
var con = mysql.createConnection({
  host: DB_HOST,     // cloudsql-proxy in prod
  user: DB_USER,     // From Kubernetes secrets in prod
  password: DB_PASS  // Null in prod
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

module.exports = {
  connect: connect,
  disconnect: disconnect 
};

/******************** Helpers *********************/

/*
 * Logs fatal error message and exists
 */
function fatal(message) {
  console.log("Fatal: " + message);
  process.exit(1);
}

/* 
 * Example usage of the exported routines
 */
function main() {
  connect();
  disconnect();
}

/*
 * If run as a script, start the server
 */
if (require.main === module) {
  main();
} 

