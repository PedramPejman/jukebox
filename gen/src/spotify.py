import spotipy

def get_top_tracks(access_token, time_range, limit=20):
  ''' Returns user's top tracks or an empty list '''

  spotify = spotipy.Spotify(auth=access_token)
  try:
    results = spotify.current_user_top_tracks(limit=limit, time_range=time_range)
    return results['items']
  except spotipy.client.SpotifyException as err:
    print("Spotify Error: {0}".format(err))
    return []

def main():
  ''' Shows example usage of routines '''
  
  token = ''
  tracks = get_top_tracks(token, 'short_term', 5)
  for track in tracks:
    print(track['name'])

if __name__=='__main__':
  main()
