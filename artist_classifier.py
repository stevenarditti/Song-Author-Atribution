import os
import json
from collections import Counter
import dataclasses
from dataclasses import dataclass
import numpy as np

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

def generate_labels(list_of_songs, list_of_artists):
    """ Save list of songs to a local json copy
        Args:
            list_of_songs(list(Song)): songs for labels to be extracted from
            list_of_artists(list(string)): artist name, each label will be an index of this list
        Returns: list of labels, (0-10) corresponding to the artist
    """
    labels = []
    for song in list_of_songs:
        labels.append(list_of_artists.index(song.artist))
    return labels


def sigmoid(x):
    """ The sigmoid function
        Arguments:
            x: input to the sigmoud function
        Return:
            sigmoid(x)
    """
    return 1 / (1 + np.exp(-1 * x))


def softmax(x):
    """ The softmax function
        Arguments:
            x: input to the softmax function
        Return:
            softmax(x)
    """
    total = np.sum([np.exp(j) for j in x])
    return [np.exp(i) / total for i in x]


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
        self.songs_per_artist = {artist: 0 for artist in self.class_labels}
        self.words_per_artist = {artist: 0 for artist in self.class_labels}
        self.total_songs = 0
        self.vocab = set()    #{artist: set() for artist in self.class_labels}
        self.vocab_size = 0   #{artist: 0 for artist in self.class_labels}

    def train(self, songs):
        for song in songs:
            artist = song.artist
            self.songs_per_artist[artist] += 1
            self.total_songs += 1
            for word in song.lyrics.lower().split():
                self.bag[artist][word] += 1
                self.words_per_artist[artist] += 1
                if word not in self.vocab:
                    self.vocab.add(word)
                    self.vocab_size += 1

    def score(self, song_lyrics):
        words = song_lyrics.split()
        probs = {}
        for artist in self.class_labels:
            prob = np.log(self.songs_per_artist[artist] / self.total_songs)
            denom = (self.words_per_artist[artist] + self.vocab_size)
            for word in words:
                if word in self.vocab:
                    prob += np.log(((self.bag[artist][word]) + 1) / denom)
            probs[str(artist)] = np.e ** prob
        return probs

    def classify(self, song_lyrics):
        scores = self.score(song_lyrics.lower())
        return max(scores, key=scores.get)


class Logistic_Regression_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "logistic_regression"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)
        self.bias = 1
        self.weights = [[1, 1, 1, 1, 1, self.bias] for _ in range(len(class_labels))]
        self.learning_rate = 0.4

    def train(self, songs):
        # Updates the classifier against given input examples
        for song in songs:
            features = self.featurize(song.lyrics)
            n = [np.dot(weight, features) for weight in self.weights]
            prob = softmax(n)
            # Calculate gradients for each weight
            idx = self.class_labels.index(song.artist)
            grads = (-np.log(prob[idx])) * np.array(features)

            # Update weights
            self.weights[idx] = self.weights[idx] - (self.learning_rate * grads)
            local_bias = self.weights[idx][-1]
            self.weights = [np.append(weight[:-1], local_bias) for weight in self.weights]
        print("Done training")

    def featurize(self, lyrics):
        # F1 number of words
        # F2 number unique words / number of words
        # F3 sentimen {0 negative, .5 neutral, 1 positive}
        # F4 nouns in song
        # F5 verbs in song
        # F6 adjectives in song
        # F7 number of named entities if it can be done "cheaply"
        return [1, 1, 1, 1, 1, 1]

    def classify(self, song_lyrics):
        features = self.featurize(song_lyrics)
        prob = softmax(np.dot(self.weights, features))
        return self.class_labels(np.argmax(prob))


class Feed_Forward_Neural_Net_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "feed_forward_neural_net"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


class Recurrent_Neural_Net_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "recurrent_neural_net"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)


# Example classifier declarations
"""
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
"""
