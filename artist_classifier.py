import os
import json
from collections import Counter
from dataclasses import dataclass
import numpy as np

from scipy.special import softmax 

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.models import Sequential
from keras.layers import Dense

STOPWORDS = stopwords.words('english')

DATA_PATH = "data"
DATA_FILE = os.path.join(DATA_PATH, "songs.json")
FEATURE_WORDS = 400
EMBEDDING_SIZE = 300
FEATURE_LENGTH = FEATURE_WORDS * EMBEDDING_SIZE
PAD = 'qqqqqqqqq'


@dataclass
class Song:
    title: str
    artist: str
    genre: str
    lyrics: str


def print_song(song):
    print(f"{song.title} by {song.artist}({song.genre}):\n{song.lyrics}\n")


def load_data(filename=None):
    """ Load list of songs from local json copy
        Ret:
            List of Song as from local data
    """
    if filename is None:
        filename = DATA_FILE
    with open(filename, 'r', encoding='utf-8') as data_file:
        return [Song(**song) for song in json.load(data_file)]


def store_data(list_of_songs, filename=None):
    """ Save list of songs to a local json copy
        Args:
            list of songs(list(Song)): songs to be saved locally
    """
    if filename is None:
        filename = DATA_FILE
    print("Saving data locally...", end="")
    with open(filename, "w+", encoding='utf-8') as data_file:
        songs = [dataclasses.asdict(song) for song in list_of_songs]
        data_file.write(json.dumps(songs))
    print("Success")


# unused in final version, included in submission
def get_label_counts(predictions, labels):
    """ gets counts of true / false positive / negative predictions
    Arguments:
        predictions: list of predicted labels
        true_labels: true labels
    Returns tuple of true positives, false positives, true negatives, and false negatives
    """
    true_pos = true_neg = false_pos = false_neg = 0

    for artist in set(labels):
        for idx, prediction in enumerate(predictions):
            if prediction == artist and prediction == labels[idx]:
                true_pos += 1
            elif prediction == artist and prediction != labels[idx]:
                false_pos += 1
            elif prediction != artist and prediction == labels[idx]:
                false_neg += 1
            elif prediction != artist and prediction != labels[idx]:
                true_neg += 1
    return true_pos, true_neg, false_pos, false_neg


# unused in final version, included in submission
def accuracy(predictions, true_labels):
    """ Overall prediction accuracy
    Arguments:
        predictions: list of predicted labels
        true_labels:      true labels
    Returns accuracy(float)
    """
    if len(predictions) != len(true_labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, true_labels)
    return true_pos / len(predictions)


# unused in final version, included in submission
def precision(predictions, true_labels):
    """ Macroaverage precision
    Arguments:
        predictions: list of predicted labels
        true_labels:      true labels
    Returns precision(float)
    """
    if len(predictions) != len(true_labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, true_labels)
    return true_pos / (true_pos+false_pos)


# unused in final version, included in submission
def recall(predictions, true_labels):
    """ Macroaverage recall
    Arguments:
        predictions: list of predicted labels
        labels:      true labels
    Returns recall(float)
    """
    if len(predictions) != len(true_labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, true_labels)
    return true_pos / (true_pos+false_neg)


# unused in final version, included in submission
def f1_score(predictions, true_labels):
    """ Macroaverage f1
    Arguments:
        predictions: list of predicted labels
        labels:      true labels
    Returns f1(float)
    """
    prec = precision(predictions, true_labels)
    rec = recall(predictions, true_labels)

    return (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


# unused in final version, included in submission
def getNounVerbAdj(sentence):
    """ Counts the number of nouns, verbs, and adjs in a sentence
        Arguments:
            sentence(string): input sentence to tag
        Return:
            Number of nouns, verbs, and adjectives in a sentence
    """
    nouns = verbs = adjs = 0

    for _, tag in pos_tag(word_tokenize(sentence)):
        if tag in ["NS", "NNS", "NNP", "NNPS"]:
            nouns += 1
        elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            verbs += 1
        elif tag in ["JJ", "JJR", "JJS"]:
            adjs += 1
    return nouns, verbs, adjs


def sigmoid(x):
    """ The sigmoid function
        Arguments:
            x: input to the sigmoid function
        Return:
            sigmoid(x)
    """
    return 1 / (1 + np.exp(-1 * x))


def lines_from_song(song_lyrics):
    ret = []
    for line in song_lyrics.split("\n"):
        if "[" not in line:
            ret.append(line)
    return ret


def preprocess_lyrics(lyrics):
    """
        Arguments:
            lyrics: lyrics to be processed
        Return:
            string, processed lyrics
    """
    processed_lyrics = ""
    for word in lyrics.lower().replace(".", "").replace(",", "").split():
        if word not in STOPWORDS:
            processed_lyrics += (word + " ")
    return processed_lyrics[0:-1]


class Artist_Classifier:
    def __init__(self, name, class_labels):
        self.name = name
        self.class_labels = class_labels
        self.vocab = set()
        self.vocab_size = 0

    def classify(self, song_lyrics):
        """ Takes lyrics and assigns the label of the most probable artist
            Args:
                song_lyrics (list of Song): Lyrics to be attributed
            Ret: The most probably artist labels
        """
        print("USE A SUBCLASS")

    def train(self, songs, labels):
        """ Train classifier on given data and update internal model
            Args:
                songs (list of Song): List of songs to train on
                labels (list of string): list of labels corresponding to songs
        """
        print("USE A SUBCLASS")

    def featurize(self, lyrics):
        word_list = lyrics.split()
        return [word_list.count(word) for word in list(self.vocab)]


    def data_generator(self, X, y, num_sequences_per_batch):
        """
        Returns data generator to be used by feed_forward
        https://wiki.python.org/moin/Generators
        https://realpython.com/introduction-to-python-generators/
        Yields batches of embeddings and labels to go with them.
        Use one hot vectors to encode the labels (see the to_categorical function)
        """
        counter = 0
        while counter < len(X):
            x_gen = []
            y_gen = []
            for i in range(counter, min(counter+num_sequences_per_batch, len(X))):
                x_gen.append(self.featurize(X[i]))
                y_vec = [0 for c in self.class_labels]
                y_vec[self.class_labels.index(y[i])] = 1
                y_gen.append(y_vec)
            yield np.array(x_gen), np.array(y_gen)
            counter += num_sequences_per_batch

    def __str__(self):
        return f"{self.name} with {len(self.class_labels)} possible artists"


class Bag_of_Words_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "bag-of-words"

    def __init__(self, name, class_labels):
        super().__init__(name, class_labels)

        self.num_artists = len(class_labels)
        self.bag = {artist: Counter() for artist in self.class_labels}
        self.songs_per_artist = {artist: 0 for artist in self.class_labels}
        self.words_per_artist = {artist: 0 for artist in self.class_labels}
        self.total_songs = 0
        self.vocab = set()
        self.vocab_size = 0


    def train(self, songs):
        # iterate through songs, gather Naive Bayes parameters
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
        self.weights = []
        self.learning_rate = .01

    def train(self, songs):
        # Updates the classifier against given input examples
        for song in songs:
            for word in song.lyrics.split():
                self.vocab.add(word)
        self.vocab_size = len(self.vocab)

        self.weights = np.random.rand(len(self.class_labels), self.vocab_size)

        for song in songs:
            features = self.featurize(song.lyrics)
            n = [np.dot(weight, features) for weight in self.weights]
            prob = softmax(n)

            # Calculate gradients for each weight
            idx = self.class_labels.index(song.artist)
            grads = (-np.log(prob[idx] + 0.00001)) * np.array(features)

            # Update weights
            self.weights[idx] = self.weights[idx] + (self.learning_rate * grads)
            local_bias = self.weights[idx][-1]
            self.weights = [np.append(weight[:-1], local_bias) for weight in self.weights]


    def classify(self, song_lyrics):
        features = self.featurize(song_lyrics)
        prob = softmax(np.dot(self.weights, features))
        return self.class_labels[np.argmax(prob)]


class Feed_Forward_Neural_Net_Artist_Classifier(Artist_Classifier):
    type_of_classifier = "feed_forward_neural_net"

    def __init__(self, name, class_labels, num_sequences_per_batch=25):
        super().__init__(name, class_labels)
        self.num_sequences_per_batch = num_sequences_per_batch
        self.nn = Sequential()


    def train(self, songs):
        for song in songs:
            for word in song.lyrics.split():
                self.vocab.add(word)

        # initialize neural net
        steps_per_epoch = len(songs) // self.num_sequences_per_batch
        self.vocab_size = len(self.vocab)
        self.nn.add(Dense(self.vocab_size, input_shape=(self.vocab_size,), activation='relu'))
        self.nn.add(Dense(len(self.class_labels), activation='softmax'))
        self.nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # generate data
        X = [song.lyrics for song in songs]
        y = [song.artist for song in songs]
        data_gen = self.data_generator(X, y, self.num_sequences_per_batch)

        # train
        self.nn.fit(x=data_gen, epochs=1, steps_per_epoch=steps_per_epoch)


    def classify(self, song_lyrics):
        features = self.featurize(song_lyrics)
        return self.class_labels[np.argmax(self.nn.predict([features]))]
