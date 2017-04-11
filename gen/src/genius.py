import pandas as pd
from sklearn import preprocessing
import numpy as np

ACOUSTICNESS = 'acousticness'
DANCEABILITY = 'danceability'
ENERGY = 'energy'
INSTRUMENTALNESS = 'instrumentalness'
LIVENESS = 'liveness'
LOUDNESS = 'loudness'
SPEECHINESS = 'speechiness'
TEMPO = 'tempo'
VALENCE = 'valence'

REQUIRED_ATTRIBUTES = [
        ACOUSTICNESS, DANCEABILITY, ENERGY,
        INSTRUMENTALNESS, LIVENESS, LOUDNESS,
        SPEECHINESS, TEMPO, VALENCE]

# TODO: Fix style issue (tab->4 spaces)
def select_best_tracks(potential_tracks, seed_tracks, feature_params):

    #unwanted_columns = ['uri', 'duration_ms', 'key', 'mode', 'time_signature']
    unwanted_columns = ['uri']
    data = raw_data.drop(unwanted_columns, axis=1)

    #seed_track_names = ['Broccoli', 'Tiimmy Turner', 'One Dance']
    seed_track_names = ["She's Mine Pt. 2", 'Wicked Games', 'Crew Love']

    seed_tracks = data.loc[data['name'].isin(seed_track_names)]
    # TODO: Get rid of literals
    seed_audio_levels = {AUDIO_FEATURE_ACOUSTICNESS:0.2, 'danceability':0.4, 'energy':0.4, 'instrumentalness':0.3,
                     'liveness':0.2, 'loudness':-2, 'speechiness':0.1, 'tempo':100, 'valence':0.3,
                     'novelty':0.5, 'popularity': 70}

    ############ process starts ##############
    # TODO: Better naming
    mapping_row = pd.Series(['test', 'test', 'test', seed_audio_levels['acousticness'], seed_audio_levels['danceability'],
                            seed_audio_levels['energy'], seed_audio_levels['instrumentalness'],
                            seed_audio_levels['liveness'], seed_audio_levels['loudness'],
                            seed_audio_levels['speechiness'], seed_audio_levels['tempo'],
                            seed_audio_levels['valence'], seed_audio_levels['popularity']],
                            index = data.columns, name='mapping_row')

    data=data.append(mapping_row)

    #scale data to have zero mean and unit variance
    for column in data.columns:
        if data[column].dtype == np.float64 or data[column].dtype == np.int64:
            data[column] = preprocessing.scale(data[column])

    # TODO: Take out
    related_artists = ['The Weeknd', 'Future', 'G-Eazy', 'Frank Ocean', 'Wiz Khalifa', 'Pusha T', 'A$AP Ferg',
                    'Maino', 'Lil Uzi Vert', 'The Notorious B.I.G', '6LACK', 'Lil Yachty', 'PnB Rock',
                    'Desiigner']

    artists = ['Drake', 'Desiigner', 'D.R.A.M.']

    related_artists_values = []

    for row in data['artist']:
        if row in related_artists or row in artists:
            related_artists_values.append(1)
        else:
            related_artists_values.append(0)

    data['related_artists'] = related_artists_values

    target_popularity = data.loc['mapping_row', 'popularity']

    popularity_deviation_values = []
    for row in data['popularity']:
        if abs(row-target_popularity)<0.5:
            popularity_deviation_values.append(3)
        elif abs(row-target_popularity)<1:
            popularity_deviation_values.append(2)
        elif abs(row-target_popularity)<1.5:
            popularity_deviation_values.append(1)
        else:
            popularity_deviation_values.append(0)

    data['popularity_deviation'] = popularity_deviation_values

    user_requested_audio_features = data.loc['mapping_row', ['acousticness','danceability', 'energy',
                                                         'instrumentalness', 'liveness', 'loudness',
                                                         'speechiness', 'tempo', 'valence']]

    distance_to_knobs=[]
    for index, row in data.iterrows():
        row_data = data.loc[index, ['acousticness','danceability', 'energy', 'instrumentalness', 'liveness',
                                  'loudness', 'speechiness', 'tempo', 'valence']]

        distance_to_knobs.append((np.linalg.norm(row_data-user_requested_audio_features)))

    data['distance_to_knobs'] = distance_to_knobs

    mean_seedtrack_values = data.loc[data['name'].isin(seed_track_names)].mean(numeric_only=True)
    mean_seedtrack_values = mean_seedtrack_values.loc[['acousticness','danceability', 'energy',
                                                         'instrumentalness', 'liveness', 'loudness',
                                                         'speechiness', 'tempo', 'valence']]

    distance_to_mean_seedtracks=[]
    for index, row in data.iterrows():
        row_data = data.loc[index, ['acousticness','danceability', 'energy', 'instrumentalness', 'liveness',
                                  'loudness', 'speechiness', 'tempo', 'valence']]

        distance_to_mean_seedtracks.append((np.linalg.norm(row_data-mean_seedtrack_values)))

    data['distance_to_seedtracks'] = distance_to_mean_seedtracks


    #scale derived features to have zero mean and unit variance
    for column in data.columns:
        if column in ['related_artists', 'popularity_deviation', 'distance_to_knobs', 'distance_to_seedtracks']:
            data[column] = preprocessing.scale(data[column])

    
    ########## Scoring starts

    data['score'] = data['related_artists'] + data['popularity_deviation'] - data['distance_to_knobs'] - data['distance_to_seedtracks']

    data.loc['mapping_row', 'score'] = -99
    data.loc[data['name'].isin(seed_track_names), 'score'] = -99

    return (data.nlargest(10, 'score'))

def create_data_frame(raw_data):
    ''' Returns DataFrame object with only the required attributes '''
    return pd.DataFrame(data=raw_data, columns=REQUIRED_ATTRIBUTES)

def pre_process(cleaned_data, initial_attributes):
    pass

def process(cleaned_data, related_artists):
    pass

def select_top_n_tracks(processed_data, n):
    pass

def select_top_tracks():
    raw_data = [
        {
            ACOUSTICNESS: .6,
            ENERGY : .6,
            'unwanted' : .7
        },
        {
            ACOUSTICNESS: .8,
            ENERGY : .7
        }]
    initial_attributes = {
            ACOUSTICNESS: .1,
            ENERGY: .9
        }
    related_artists = []
    n = 10

    df = create_data_frame(raw_data)
    preprocessed_data = pre_process(df, initial_attributes)

    processed_data = process(cleaned_data, related_artists)
    selected_tracks = select_top_n_tracks(processed_data, n)

def main():
    # clean data
    # prepocess data
    # compute derivative attributes
    # compute score
    
    select_top_tracks()


if __name__ == "__main__":
    main()
    # raw_data = pd.read_csv('test_data_150_top_songs_single_artist.csv')
    # prepared_data = prepare_data(raw_data)
    # (raw_data)

