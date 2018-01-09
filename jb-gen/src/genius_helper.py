import pandas as pd
import numpy as np
import warnings

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
POPULARITY_DEVIATION_LOW = 0.7
POPULARITY_DEVIATION_MEDIUM = 1.2
POPULARITY_DEVIATION_HIGH = 1.8
POPULARITY_SIGNAL_STRONG = 1
POPULARITY_SIGNAL_MEDIUM = 0.6
POPULARITY_SIGNAL_WEAK = 0.4
POPULARITY_SIGNAL_ZERO = 0

# Regression coefficients
COEFFICIENT_RELATED_ARTISTS = .6
COEFFICIENT_POPULARITY_SIGNAL = 1
COEFFICIENT_AUDIO_FEATURE_DEVIATION = -2
COEFFICIENT_SEED_TRACK_DEVIATION = -2

REQUIRED_ATTRIBUTES = [
        URI, NAME, ALBUM, POPULARITY, ARTIST,
        ACOUSTICNESS, DANCEABILITY, ENERGY, TEMPO, VALENCE]

AUDIO_FEATURE_ATTRIBUTES = [
        ACOUSTICNESS, ENERGY, DANCEABILITY, TEMPO, VALENCE
    ]

DERIVED_ATTRIBUTES = [
        RELATED_ARTISTS, POPULARITY_SIGNAL, 
        AUDIO_FEATURE_DEVIATION]

INITIAL_ATTRIBUTES = 'initial-attributes'
NONE = 'none'
NUM_OUTPUT_TRACKS = 10 

def create_data_frame(raw_data):
    ''' Returns DataFrame object with only the required attributes '''
    return pd.DataFrame(data=raw_data, columns=REQUIRED_ATTRIBUTES)

def pre_process(df, initial_attributes, seed_tracks):
    ''' Returns DataFrame object with normalized attributes. Also
    appends the initial attributes to the dataframe for uniform normalization.'''

    # Append seed tracks 
    seed_tracks_row = pd.DataFrame(data=seed_tracks,
        columns = REQUIRED_ATTRIBUTES)
    df = df.append(seed_tracks_row, ignore_index=True)

    # Append initial attributes
    initial_attr_row = pd.Series(initial_attributes, 
        index=df.columns, name=INITIAL_ATTRIBUTES)
    initial_attr_row = initial_attr_row.fillna(value=NONE)
    df = df.append(initial_attr_row)

    # Drop rows with missing data
    df = df.dropna(how='any')

    # Drop duplicates
    df = df.drop_duplicates(URI)

    # Raise error if there are no tracks left
    if (df.size == 0):
        raise ValueError("Cannot normalize empty dataframe")

    # Normalize data to have zero mean and unit variance
    for column in df.columns:
        if (df[column].dtype == np.float64 or df[column].dtype == np.int64):
            df[column] = scale(df[column])

    return df
    
def process(df, seed_tracks, related_artists):
    ''' Returns dataframe with derived columns computed '''
    
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
        elif (deviation < POPULARITY_DEVIATION_MEDIUM):
            popularity_deviation_values.append(POPULARITY_SIGNAL_MEDIUM)
        elif (deviation < POPULARITY_DEVIATION_HIGH):
            popularity_deviation_values.append(POPULARITY_SIGNAL_WEAK)
        else: 
            popularity_deviation_values.append(POPULARITY_SIGNAL_ZERO)
    df[POPULARITY_SIGNAL] = popularity_deviation_values

    # Compute AUDIO_FEATURE_DEVIATION
    target_features = df.loc[INITIAL_ATTRIBUTES, AUDIO_FEATURE_ATTRIBUTES]
    audio_feature_deviations=[]
    for index, row in df.iterrows():
        row_data = df.loc[index, AUDIO_FEATURE_ATTRIBUTES]
        audio_feature_deviations.append(
                (np.linalg.norm(row_data - target_features)))
    df[AUDIO_FEATURE_DEVIATION] = audio_feature_deviations
    
    # Extract mean seedtrack values
    seed_track_uris = [track[URI] for track in seed_tracks]
    seed_tracks = df.loc[df[URI].isin(seed_track_uris), AUDIO_FEATURE_ATTRIBUTES]
    mean_seedtrack_values = seed_tracks.mean(numeric_only=True)

    # Compute MEAN_SEED_TRACKS_DEVIATION
    distance_to_mean_seedtracks=[]
    for index, row in df.iterrows():
        row_data = df.loc[index, AUDIO_FEATURE_ATTRIBUTES]
        distance_to_mean_seedtracks.append(
                (np.linalg.norm(row_data - mean_seedtrack_values)))
    df[MEAN_SEED_TRACK_DEVIATION] = distance_to_mean_seedtracks

    # Normalize derived attributes, ignoring float<->int conversion warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attribute in DERIVED_ATTRIBUTES:
            df[attribute] = scale(df[attribute])
    
    # Take out seed tracks and initial attributes rows
    df.drop(INITIAL_ATTRIBUTES, inplace=True)
    df.drop(df[URI].isin(seed_track_uris), inplace=True)

    return df
    
def select_top_n_tracks(df, n):
    ''' Returns the top n tracks from the dataframe based on score attribute '''

    df[SCORE] = COEFFICIENT_RELATED_ARTISTS * df[RELATED_ARTISTS] + \
        COEFFICIENT_POPULARITY_SIGNAL * df[POPULARITY_SIGNAL] + \
        COEFFICIENT_AUDIO_FEATURE_DEVIATION * df[AUDIO_FEATURE_DEVIATION] + \
        COEFFICIENT_SEED_TRACK_DEVIATION * df[MEAN_SEED_TRACK_DEVIATION] 
                    
    return (df.nlargest(n, SCORE))
