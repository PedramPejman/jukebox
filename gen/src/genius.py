import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
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

'''REQUIRED_ATTRIBUTES = [
        URI, NAME, ALBUM, POPULARITY, ARTIST,
        ACOUSTICNESS, DANCEABILITY, ENERGY,
        INSTRUMENTALNESS, LIVENESS, LOUDNESS,
        SPEECHINESS, TEMPO, VALENCE]
'''
REQUIRED_ATTRIBUTES = [
        URI, POPULARITY, ARTIST, ENERGY, ACOUSTICNESS
    ]
AUDIO_FEATURE_ATTRIBUTES = [
        ACOUSTICNESS, ENERGY
    ]
DERIVED_ATTRIBUTES = [
        RELATED_ARTISTS, POPULARITY_SIGNAL, 
        AUDIO_FEATURE_DEVIATION]

INITIAL_ATTRIBUTES = 'initial-attributes'
NONE = 'none'

# Derived attribute names
RELATED_ARTISTS = 'related-artists'
POPULARITY_SIGNAL = 'populatiry-signal'
AUDIO_FEATURE_DEVIATION = 'audio-feature-deviation'
MEAN_SEED_TRACK_DEVIATION = 'seed-track-deviation'
SCORE = 'score'

# Derived attributes parameters
RELATED_ARTISTS_STRONG = 2
RELATED_ARTISTS_MEDIUM = 1
RELATED_ARTISTS_WEAK = 0
POPULARITY_SIGNAL_LOW = .2
POPULARITY_SIGNAL_MEDIUM = .4
POPULARITY_SIGNAL_HIGH = .6
POPULARITY_SIGNAL_STRONG = 1
POPULARITY_SIGNAL_MEDIUM = 0.6
POPULARITY_SIGNAL_WEAK = 0.4
POPULARITY_SIGNAL_ZERO = 0

# Regression coefficients
COEFFICIENT_RELATED_ARTISTS = 1
COEFFICIENT_POPULARITY_SIGNAL = 1
COEFFICIENT_AUDIO_FEATURE_DEVIATION = -1 
COEFFICIENT_SEED_TRACK_DEVIATION = -1

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
    ''' Returns DataFrame object with normalized attributes. Also
    appends the initial attributes to the dataframe for uniform normalization.'''

    # Append initial attributes
    initial_attr_row = pd.Series(initial_attributes, 
        index=cleaned_df.columns, name=INITIAL_ATTRIBUTES)
    initial_attr_row = initial_attr_row.fillna(value=NONE)
    df = cleaned_df.append(initial_attr_row)

    # Drop rows with missing data
    df = df.dropna(how='any')

    # Raise error if there are no tracks left
    if (df.size == 0):
        raise ValueError("Cannot normalize empty dataframe")

    # Normalize data to have zero mean and unit variance
    for column in df.columns:
        if (df[column].dtype == np.float64 or df[column].dtype == np.int64):
            df[column] = scale(df[column])

    return df
    
def process(df, seed_tracks, related_artists):
    
    # Compute set of artsts
    artists = set([track[ARTIST] for track in seed_tracks])
    
    # Compute RELATED_ARTISTS attribute
    related_artists_values = []
    for row in df[ARTIST]:
        if row in artists:
            related_artists_values.append(RELATED_ARTISTS_STRONG)
        elif row in related_artists:
            related_artists_values.append(RELATED_ARTISTS_MEDIUM)
        else:
            related_artists_values.append(RELATED_ARTISTS_WEAK)
    df[RELATED_ARTISTS] = related_artists_values

    # Compute POPULARITY_SIGNAL attribute
    target_popularity = df.loc[INITIAL_ATTRIBUTES, POPULARITY]
    popularity_deviation_values = []
    for row in df[POPULARITY]:
        deviation = abs(row - target_popularity)
        if (deviation < POPULARITY_DEVIATION_LOW):
            popularity_deviation_values.append(POPULARITY_SIGNAL_STRONG)
        elif (deviation < POPULARITY_SIGNAL_MEDIUM):
            popularity_deviation_values.append(POPULARITY_SIGNAL_MEDIUM)
        elif (deviation < POPULARITY_SIGNAL_HIGH):
            popularity_deviation_values.append(POPULARITY_SIGNAL_WEAK)
        else: 
            popularity_deviation_values.append(POPULARITY_SIGNAL_ZERO)
    df[POPULARITY_SIGNAL] = popularity_deviation_values

    # Compute AUDIO_FEATURE_DEVIATION
    target_features = data.loc[INITIAL_ATTRIBUTES, AUDIO_FEATURE_ATTRIBUTES]
    audio_feature_deviations=[]
    for index in df.iterrows():
        row_data = df.loc[index, AUDIO_FEATURE_ATTRIBUTES]
        audio_feature_deviations.append(
                (np.linalg.norm(row_data - user_requested_audio_features)))
    df[AUDIO_FEATURE_DEVIATION] = audio_feature_deviations
    
    # Extract mean seedtrack values
    seed_track_uris = [track[URI] for track in seed_tracks]
    mean_seedtrack_values = df.loc[df[URI].isin(seed_track_uris)].mean(numeric_only=True)

    # Compute MEAN_SEED_TRACKS_DEVIATION
    distance_to_mean_seedtracks=[]
    for index in df.iterrows():
        row_data = df.loc[index, REQUIRED_ATTRIBUTES]
        distance_to_mean_seedtracks.append(
                (np.linalg.norm(row_data - mean_seedtrack_values)))
    df[MEAN_SEED_TRACKS_DEVIATION] = distance_to_mean_seedtracks

    # Normalize derived attributes
    for attribute in DERIVED_ATTRIBUTES:
        df[attribute] = scale(df[attribute])

    return df
    
def select_top_n_tracks(processed_data, n):
    df[SCORE] = COEFFICIENT_RELATED_ARTISTS * df[RELATED_ARTISTS] +
        COEFFICIENT_POPULARITY_SIGNAL * df[POPULARITY_SIGNAL] +
        COEFFICIENT_AUDIO_FEATURE_DEVIATION * df[AUDIO_FEATURE_DEVIATION] +
        COEFFICIENT_SEED_TRACK_DEVIATION * df[MEAN_SEED_TRACKS_DEVIATION] 
                    
    # Remove INITIAL_ATTRIBUTES and seed tracks
    data.loc['mapping_row', 'score'] = -99
    data.loc[data['name'].isin(seed_track_names), 'score'] = -99

    return (data.nlargest(10, 'score'))

def select_top_tracks():
    raw_data = [
        {
            URI : 'uri10',
            ARTIST : 'artist3',
            ACOUSTICNESS : .6,
            ENERGY : .6,
            POPULARITY : 30,
            'unwanted' : .7
        },
        {
            URI : 'uri11',
            ARTIST : 'artist1',
            ACOUSTICNESS: .8,
            ENERGY : .7,
            POPULARITY : 80,
        }]
    initial_attributes = {
            ACOUSTICNESS: .1,
            POPULARITY : 20,
            ENERGY: .9
        }
    seed_tracks = [
        {
            URI : 'uri1',
            ARTIST: 'artist1',
            ACOUSTICNESS : .3,
            ENERGY : .4,
            POPULARITY : 50,
        },
        {
            URI : 'uri2',
            ARTIST : 'artist2',
            ACOUSTICNESS : .3,
            ENERGY : .6,
            POPULARITY : 20,
        }]
    related_artists = ['artist1']
    n = 10

    cleaned_df = create_data_frame(raw_data)
    preprocessed_df = pre_process(cleaned_df, initial_attributes)
    print(preprocessed_df)
    processed_data = process(preprocessed_df, seed_tracks, related_artists)
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

