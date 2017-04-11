import pandas as pd
import numpy as np

from sklearn.prepocessing import scale
from sklearn.preprocessing import Imputer

# Track identifiers
URI = 'uri'
NAME = 'name'
ALBUM = 'album'
POPULARITY = 'popularity'
ARTIST = 'artist'

# Track audio features
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
        URI, NAME, ALBUM, POPULARITY, ARTIST,
        ACOUSTICNESS, DANCEABILITY, ENERGY,
        INSTRUMENTALNESS, LIVENESS, LOUDNESS,
        SPEECHINESS, TEMPO, VALENCE]

# Derived attribute names
RELATED_ARTISTS = 'related-artists'

# Derived attributes parameters
RELATED_ARTISTS_STRONG = 2
RELATED_ARTISTS_MEDIUM = 1
RELATED_ARTISTS_WEAK = 0

# TODO: Fix style issue (tab->4 spaces)
def select_best_tracks(potential_tracks, seed_tracks, feature_params):

    #seed_track_names = ['Broccoli', 'Tiimmy Turner', 'One Dance']
    seed_track_names = ["She's Mine Pt. 2", 'Wicked Games', 'Crew Love']

    seed_tracks = data.loc[data['name'].isin(seed_track_names)]
    # TODO: Get rid of literals
    seed_audio_levels = {AUDIO_FEATURE_ACOUSTICNESS:0.2, 'danceability':0.4, 'energy':0.4, 'instrumentalness':0.3,
                     'liveness':0.2, 'loudness':-2, 'speechiness':0.1, 'tempo':100, 'valence':0.3,
                     'novelty':0.5, 'popularity': 70}


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

def pre_process(cleaned_df, initial_attributes):
    ''' Returns DataFrame object with normalized attributes '''
    target_df = pd.DataFrame(initial_attributes, [0], columns=REQUIRED_ATTRIBUTES)
    df = cleaned_df.append(target_df, ignore_index=True)

    # Impute missing values
    imp = Imputer(

    # Scale data to have zero mean and unit variance
    for column in df.columns:
        if (df[column].dtype == np.float64 or df[column].dtype == np.int64) and df[column]:
            print(column, end=" ")
            print(df[column])
            df[column] = preprocessing.scale(df[column])

    return df
    
def process(df, seed_tracks, related_artists):
    
    
    # Compute set of artsts
    artists = set([track[ARTIST] for track in seed_tracks])
    
    related_artists_values = []
    for row in df[ARTIST]:
        if row in artists:
            related_artists_values.append(RELATED_ARTISTS_STRONG)
        elif row in related_artists:
            related_artists_values.append(RELATED_ARTISTS_MEDIUM)
        else:
            related_artists_values.append(RELATED_ARTISTS_WEAK)

    df[RELATED_ARTISTS] = related_artists_values
    print(artists)
    print(df)

def select_top_n_tracks(processed_data, n):
    pass

def select_top_tracks():
    raw_data = [
        {
            ARTIST : 'artist3',
            ACOUSTICNESS : .6,
            ENERGY : .6,
            'unwanted' : .7
        },
        {
            ARTIST : 'artist1',
            ACOUSTICNESS: .8,
            ENERGY : .7
        }]
    initial_attributes = {
            ACOUSTICNESS: .1,
            ENERGY: .9
        }
    seed_tracks = [
        {
            URI : 'uri1',
            ARTIST: 'artist1',
            ENERGY : .4
        },
        {
            URI : 'uri2',
            ARTIST : 'artist2',
            ENERGY : .6
        }]
    related_artists = ['artist1']
    n = 10

    cleaned_df = create_data_frame(raw_data)
    preprocessed_df = pre_process(cleaned_df, initial_attributes)
    processed_data = process(cleaned_df, seed_tracks, related_artists)
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

