import lyricsgenius
import json


DEBUG = True    # print debug statements
MAX_SONGS = 50  # number of songs to pull per artist
ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin']


def __debug(s):
    if DEBUG:
        print(s)


def __get_session():
    access_token = 'RMYdtaY1jKu2KArUDJXDV5aRh17IjVXLuRjZt7qm9Z17gg2ayLwOQYnxqqRmIAhh'
    return lyricsgenius.Genius(access_token)


def get_lyric_data():

    session = __get_session()

    lyrics_json = {}
    for artist in ARTISTS:
        artist_result = session.search_artist(artist, max_songs=MAX_SONGS, sort="popularity")

        lyrics = []
        for song in artist_result.songs:
            lyrics.append({song.title: song.lyrics})
        
        lyrics_json[artist] = lyrics
    
    f = open('lyrics.json', 'w')
    json.dump(lyrics_json, f)
    f.close()
