import os
import json
import dataclasses
from dataclasses import dataclass

DATA_PATH = "data"
DATA_FILE = os.path.join(DATA_PATH, "songs.json")


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

    def classify(song_lyrics):
        """ Takes lyrics and assigns the label of the most probable artist
            Args:
                song_lyrics (TODO: Format lyrics): Lyrics to be attributed
            Ret: The most probably artist labels
        """
        print("USE A SUBCLASS")

    def train(songs, labels):
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

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


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
a = Bag_of_Words_Artist_Classifier("Bag-of-words general", ["Kendrick Lamar", "The Beatles", "Led Zeplin"])
b = Logistic_Regression_Artist_Classifier("LogRes for pop", ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin'])
c = Feed_Forward_Neural_Net_Artist_Classifier("FFNN for country", ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin'])
d = Recurrent_Neural_Net_Artist_Classifier("RNN for rap", ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin'])
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
