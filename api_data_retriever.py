from rauth import OAuth2Service


DEBUG = True    # print debug statements


def debug(s):
    if DEBUG:
        print(s)


# gets a new OAuth2 session used for API calling.
# call at begininng of script and use for all API retrievals
def get_session():
    client_id = 'r3aOp51BYzVDSM6Pyjtr9YJj2VTUvPDgzaZqIsNmbnEWBMTUX0Ve-bJRtgXuOE8G'
    client_not_so_secret = 'XxYFdUG5wu5elBKZcabvqHjjd5E-bSCLT5xso1xc-NrtsUMarepeEDBZQPd6xvc1VRgrxykpQ'
    access_token = 'RMYdtaY1jKu2KArUDJXDV5aRh17IjVXLuRjZt7qm9Z17gg2ayLwOQYnxqqRmIAhh'

    service = OAuth2Service(client_id=client_id,
                            client_secret=client_not_so_secret,
                            authorize_url='https://api.genius.com/oauth/authorize',
                            access_token_url='https://api.genius.com/oauth/token',
                            base_url='https://api.genius.com')

    return service.get_session(access_token)


# retrieves song data for given API song id
def get_song(session, id):
    url = 'songs/' + id
    debug(url)
    return session.get(url, params={'format': 'json'}).json()['response']['song']


# finds an artist's API id by pulling the top search result
def find_artist_id(session, name):
    search = session.get('search', params={'q': name, 'format': 'json'}).json()
    # pull artist from first search result
    # TODO, try catch? need to ensure this gives what we need
    artist = search['response']['hits'][0]['result']['primary_artist']

    return artist['name'], artist['api_path']


# creates a list of artist IDs from a list of artist names
def get_artist_paths(session, artists):
    paths = []
    for artist in artists:
        name, path = find_artist_id(session, artist)

        # if there is a name discrepancy, it may have pulled the wrong artist
        if name != artist:
            print(f"Difference in API name:\n{artist} != {name}")
        paths.append(path)
    return paths


def main():
    artist_list = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin']

    session = get_session()

    # retrieve artist API ids
    artist_api_paths = get_artist_paths(session, artist_list)

    print(artist_api_paths)


if __name__ == '__main__':
    main()
