from genius_helper import *
from spotify import *

def get_seed_tracks(access_token):
    ''' Returns potential seed tracks for user given access token '''

    # TODO: Thread these 3 calls
    st_tracks = get_top_tracks(access_token, 'short_term')
    mt_tracks = get_top_tracks(access_token, 'medium_term')
    lt_tracks = get_top_tracks(access_token, 'long_term')

    tracks = []
    tracks.extend(st_tracks[0:2])
    tracks.extend(mt_tracks[0:6])
    tracks.extend(lt_tracks[0:2])

    return tracks

def select_top_tracks(raw_data, initial_attributes, 
    seed_tracks, related_artists, n=NUM_OUTPUT_TRACKS):
    ''' Returns top n tracks from raw_data '''

    # Clean tracks dataframe and strip unnecessary attributes 
    cleaned_df = create_data_frame(raw_data)

    # Normalize data, append seed tracks and initial params
    try:
      preprocessed_df = pre_process(cleaned_df, initial_attributes, seed_tracks)
    except ValueError as err:
      print("Encountered ValueError during preprocessing: {0}".format(err))
      return []

    # Compute derived columns and assign a score to each track
    processed_data = process(preprocessed_df, seed_tracks, related_artists)

    # Select top n tracks based on a linear regression
    selected_tracks = select_top_n_tracks(processed_data, n)

    return selected_tracks.loc[:, URI].tolist()

def main():
    ''' Example usage of select_top_tracks '''

    # Initial jukebox attributes, set by the user
    initial_attributes = {
            ACOUSTICNESS: .1,
            POPULARITY : 40,
            ENERGY: .9,
            VALENCE: .3,
            TEMPO: 100,
            DANCEABILITY: .4
        }

    # Jukebox seed tracks, selected by the user
    seed_tracks = [
        {
            URI : 'spotify:track:2fkeWbM6iqTw7oGHTYm2lw',
            ARTIST: 'Drake',
            ACOUSTICNESS : .5,
            ENERGY : .24,
            POPULARITY : 83,
            TEMPO: 100,
            VALENCE: .4,
            DANCEABILITY: .2
        },
        {
            URI : 'spotify:track:1mhVXWduD8ReDwusfaHNwU',
            ARTIST : 'The Weeknd',
            ACOUSTICNESS : .25,
            ENERGY : .5,
            POPULARITY : 68,
            TEMPO: 110,
            VALENCE: .6,
            DANCEABILITY: .7
        }]

    # Related artists, computed by get_related_artists(seed_tracks)
    # TODO: implement get_related_artists()
    related_artists = ['The Weeknd', 'Future', 'G-Eazy', 'Frank Ocean', 'Wiz Khalifa', 'Pusha T', 'A$AP Ferg']

    # User's top songs, computed by get_user_top_tracks(access_token)
    # TODO: implement get_user_top_tracks
    FILE = '/usr/jukebox/gen/src/test_data_150_top_songs_single_artist.csv'
    top_songs = pd.read_csv(FILE)

    top_tracks = select_top_tracks(top_songs, initial_attributes, seed_tracks, related_artists)
    print(top_tracks)


if __name__ == "__main__":
    main()
    # raw_data = pd.read_csv('test_data_150_top_songs_single_artist.csv')
    # prepared_data = prepare_data(raw_data)
    # (raw_data)

