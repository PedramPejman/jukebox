-- Create the development database
CREATE DATABASE jb_dev;

--Use development database
USE jb_dev;

/* Create User table
+-------------------+--------------+------+-----+-------------------+-----------------------------+
| Field             | Type         | Null | Key | Default           | Extra                       |
+-------------------+--------------+------+-----+-------------------+-----------------------------+
| id                | varchar(128) | NO   | PRI | NULL              |                             |
| access_token      | varchar(512) | NO   |     | NULL              |                             |
| creation_time     | datetime     | YES  |     | CURRENT_TIMESTAMP |                             |
| modification_time | datetime     | YES  |     | NULL              | on update CURRENT_TIMESTAMP |
+-------------------+--------------+------+-----+-------------------+-----------------------------+
*/
CREATE TABLE IF NOT EXISTS user (
  `id` VARCHAR(128) NOT NULL PRIMARY KEY, 
  `access_token` VARCHAR(512) NOT NULL, 
  `creation_time` DATETIME DEFAULT CURRENT_TIMESTAMP, 
  `modification_time` DATETIME ON UPDATE CURRENT_TIMESTAMP);

/* Create TopTracks table
+------------------+--------------+------+-----+-------------------+-------+
| Field            | Type         | Null | Key | Default           | Extra |
+------------------+--------------+------+-----+-------------------+-------+
| user_id          | varchar(128) | NO   | PRI | NULL              |       |
| track_uri        | varchar(128) | NO   | PRI | NULL              |       |
| num_times_seeded | smallint(8)  | YES  |     | 0                 |       |
| creation_time    | datetime     | YES  |     | CURRENT_TIMESTAMP |       |
+------------------+--------------+------+-----+-------------------+-------+
*/
CREATE TABLE IF NOT EXISTS top_track (
  `user_id` VARCHAR(128) NOT NULL, 
  `track_uri` VARCHAR(128) NOT NULL,
  `num_times_seeded` SMALLINT(8) DEFAULT 0,
  `creation_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`, `track_uri`),
  FOREIGN KEY (`user_id`) REFERENCES user(`id`)
    ON DELETE CASCADE);

/* Create TrackFeatures table
+--------------+--------------+------+-----+---------+-------+
| Field        | Type         | Null | Key | Default | Extra |
+--------------+--------------+------+-----+---------+-------+
| track_uri    | varchar(128) | NO   | PRI | NULL    |       |
| acousticness | float        | YES  |     | NULL    |       |
| danceability | float        | YES  |     | NULL    |       |
| energy       | float        | YES  |     | NULL    |       |
| tempo        | float        | YES  |     | NULL    |       |
| valence      | float        | YES  |     | NULL    |       |
+--------------+--------------+------+-----+---------+-------+
*/
CREATE TABLE IF NOT EXISTS track_features (
  `track_uri` VARCHAR(128) NOT NULL PRIMARY KEY,
  `acousticness` FLOAT,
  `danceability` FLOAT,
  `energy` FLOAT,
  `tempo` FLOAT,
  `valence` FLOAT,
  FOREIGN KEY (`track_uri`) REFERENCES top_track(`track_uri`)
    ON DELETE CASCADE); 

/* Create Playlist table
*/
CREATE TABLE IF NOT EXISTS playlist (
  `id` INTEGER PRIMARY KEY AUTO_INCREMENT,
  `userId` VARCHAR(128) NOT NULL, 
  `uri` VARCHAR(64) NOT NULL,
  `creation_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`userId`) REFERENCES user(`id`)
    ON DELETE CASCADE);
