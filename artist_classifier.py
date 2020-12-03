import os
import json
from collections import Counter
import dataclasses
from dataclasses import dataclass

DEBUG = True
DATA_PATH = "data"
DATA_FILE = os.path.join(DATA_PATH, "songs.json")


def debug(s):
    if DEBUG:
        print(s)


@dataclass
class Song:
    title: str
    artist: str
    genre: str
    lyrics: str


def print_song(song):
    print(f"{song.title} by {song.artist}({song.genre}):\n{song.lyrics}\n")


def load_data():
    """ Load list of songs from local json copy
        Ret:
            List of Song as from local data
    """
    with open(DATA_FILE, 'r', encoding='utf-8') as data_file:
        return [Song(**song) for song in json.load(data_file)]


def store_data(list_of_songs):
    """ Save list of songs to a local json copy
        Args:
            list of songs(list(Song)): songs to be saved locally
    """
    print("Saving data locally...", end="")
    with open(DATA_FILE, "w+", encoding='utf-8') as data_file:
        songs = [dataclasses.asdict(song) for song in list_of_songs]
        data_file.write(json.dumps(songs))
    print("Success")


class Artist_Classifier:
    def __init__(self, name, class_labels):
        self.name = name
        self.class_labels = class_labels

    def classify(self, song_lyrics):
        """ Takes lyrics and assigns the label of the most probable artist
            Args:
                song_lyrics (TODO: Format lyrics): Lyrics to be attributed
            Ret: The most probably artist labels
        """
        print("USE A SUBCLASS")

    def train(self, songs, labels):
        """ Train classifier on given data and update internal model
            Args:
                songs (TODO: Format lyrics): List of songs to train on
                labels (list of labesl): list of labels corresponding to songs
        """
        print("USE A SUBCLASS")

    def __str__(self):
        return f"{self.name} with {len(self.class_labels)} possible artists"


class Bag_of_Words_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "bag-of-words"

    # self.bag:     bag of words, 1 per class
    # counts of total examples for class
    # counts of total words for each class
    # total examples
    # vocab
    # vocab size


    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)

        self.num_artists = len(class_labels)
        self.bag = {artist: Counter() for artist in self.class_labels}
        self.songs_per_artist = {0 for artist in self.class_labels}
        self.words_per_artist = {0 for artist in self.class_labels}
        self.total_songs = 0
        self.vocab = {set() for artist in self.class_labels}
        self.vocab_sizes = {0 for artist in self.class_labels}

    
    def train(self, songs, labels):
        for song in songs:
            artist = songs.artist
            self.songs_per_artist[artist] += 1
            self.total_songs += 1
            for word in song.lyrics.split():
                self.bag[artist][word] += 1
                self.words_per_artist[artist] += 1
                if word not in self.vocab[artist]:
                    self.vocab[artist].append(word)
                    self.vocab_size[artist] += 1

    def classify(self, song_lyrics):
        pass





class Logistic_Regression_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "logistic_regression"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


class Feed_Forward_Neural_Net_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "feed_forward_neural_net"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


class Recurrent_Neural_Net_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "recurrent_neural_net"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


# Example classifier declarations
class_labels = ["Kendrick Lamar", "The Beatles", "Led Zeplin"]
a = Bag_of_Words_Artist_Classifier("Bag-of-words general", )
b = Logistic_Regression_Artist_Classifier("LogRes for pop", class_labels)
c = Feed_Forward_Neural_Net_Artist_Classifier("FFNN for country", class_labels)
d = Recurrent_Neural_Net_Artist_Classifier("RNN for rap", class_labels)
print(a)
print(b)
print(c)
print(d)
print("\n\n")

# Test the loading and storing of data
x = Song("King Kunta", "Kendrick Lamar", "Rap", "I got a bone to pick...")
y = Song("Yesterday", "The Beatles", "Rock", "All my troubles seemed so far away...")
z = Song("Stairway to Heaven", "Led Zepplin", "Rock", "There's a lady who's sure. All that glitters is gold...")
songs = [x, y, z]

print_song(x)

store_data(songs)
print(songs)
songs = load_data()
print(songs)
