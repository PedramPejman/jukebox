var mysql = require('mysql');

var DB_USER = process.env.DB_USER || fatal("DB_USER not an environment variable");
var PROXY_HOST = '127.0.0.1';

// Create a connection to the Mysql proxy server
var con = mysql.createConnection({
  host: PROXY_HOST,  // Using cloudsql-proxy
  user: DB_USER      // From Kubernetes secrets
});

/****************** Exported routines *****************/

/*
 * Connect to the database server
 */
function connect() {
  con.connect(function(err){
    if(err){
      console.log('Error connecting to Db: ' + err);
      return;
    }
    console.log('Connection established');
  });
}

/*
 * Disconnect from the database server
 */
function disconnect() {
  con.end(function(err) {
    if (err) {
      console.log("Could not end connection: " + err);
      return;
    }
    console.log("Ended connection");
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

