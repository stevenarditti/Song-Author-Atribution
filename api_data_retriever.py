import lyricsgenius
from artist_classifier import Song


DEBUG = True    # print debug statements
MAX_SONGS = 150  # number of songs to pull per artist
ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin', "Aerosmith", "Frank Sinatra", "Kanye West", "Eminem", "Red Hot Chili Peppers", "Queen", "Billy Joel", "Madonna", "Drake"]


def __debug(s):
    if DEBUG:
        print(s)


def __get_session():
    access_token = 'RMYdtaY1jKu2KArUDJXDV5aRh17IjVXLuRjZt7qm9Z17gg2ayLwOQYnxqqRmIAhh'
    return lyricsgenius.Genius(access_token)


def get_lyric_data(artist):
    session = __get_session()
    output_songs = []

    artist_result = session.search_artist(artist, max_songs=MAX_SONGS, sort="popularity")

    for song in artist_result.songs:
        output_songs.append(Song(song.title, artist, None, song.lyrics))

    return output_songs
