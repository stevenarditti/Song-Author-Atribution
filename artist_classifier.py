import os
import json
from collections import Counter
import dataclasses
from dataclasses import dataclass
import numpy as np
import statistics as stats
import spacy

from nltk.corpus import stopwords
from nltk import pos_tag, download
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Embedding

download('vader_lexicon')
download('averaged_perceptron_tagger')
STOPWORDS = stopwords.words('english')
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

DEBUG = True
DATA_PATH = "data"
DATA_FILE = os.path.join(DATA_PATH, "songs.json")
FEATURE_WORDS = 400
EMBEDDING_SIZE = 300
FEATURE_LENGTH = FEATURE_WORDS * EMBEDDING_SIZE 
PAD = 'qqqqqqqqq'


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


def get_label_counts(predictions, labels):
    true_pos, true_neg, false_pos, false_neg = 0

    for class_label in range(max(labels)):
        for idx, prediction in enumerate(predictions):
            if prediction == class_label and labels[idx] == class_label:
                true_pos += 1
            elif prediction == class_label and labels[idx] != class_label:
                false_pos += 1
            elif prediction != class_label and labels[idx] == class_label:
                false_neg += 1
            elif prediction != class_label and labels[idx] != class_label:
                true_neg += 1
    return true_pos, true_neg, false_pos, false_neg


def accuracy(predictions, labels):
    """ Overall prediction accuracy
    Arguments:
        predictions: list of predicted labels
        labels:      true labels
    Returns accuracy(float)
    """
    if len(predictions) != len(labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, labels)
    return true_pos / len(predictions)


def precision(predictions, labels):
    """ Macroaverage precision
    Arguments:
        predictions: list of predicted labels
        labels:      true labels
    Returns precision(float)
    """
    if len(predictions) != len(labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, labels)
    return true_pos / (true_pos+false_pos)


def recall(predictions, labels):
    """ Macroaverage recall
    Arguments:
        predictions: list of predicted labels
        labels:      true labels
    Returns recall(float)
    """
    if len(predictions) != len(labels):
        print("Predictions and labels are different sizes. Exiting...")
        return

    true_pos, true_neg, false_pos, false_neg = get_label_counts(predictions, labels)
    return true_pos / (true_pos+false_neg)


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


def lines_from_song(song_lyrics):
    ret = []
    for line in song_lyrics.split("\n"):
        if "[" not in line:
            ret.append(line)
    return ret


def preprocess_lyrics(lyrics):
    """stop_words = [".", ",", "\n", "a", "the", "is", "i", "am", "are"]
    lyrics = lyrics.lower()
    for word in stop_words:
        lyrics = lyrics.replace(word, "")
    return lyrics"""
    processed_lyrics = ""
    for word in lyrics.lower().replace(".", "").replace(",", "").split():
        if word not in STOPWORDS:
            processed_lyrics += (word + " ")
    return processed_lyrics[0:-1]



class Artist_Classifier:
    def __init__(self, name, class_labels):
        self.name = name
        self.class_labels = class_labels
        self.embeddings = spacy.load("en_core_web_md")

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

    """
    def featurize(self, lyrics):
        # F1 number of words
        num_words = len(lyrics) + 1
        #return [np.log(num_words)]

        # F2 number unique words / number of words
        unique_words = set()
        for word in lyrics:
            unique_words.add(word)
        num_unique_words = len(unique_words) + 1


        # F3 sentiment {0 negative, .5 neutral, 1 positive}
        sentiment = SENTIMENT_ANALYZER.polarity_scores(lyrics)["pos"]

        # F4 nouns in song
        # F5 verbs in song
        # F6 adjectives in song
        noun, verb, adj = getNounVerbAdj(lyrics)

        # F7 number of named entities if it can be done "cheaply" PROBABLY NOT
        #print([np.log(num_words), num_unique_words/num_words, sentiment, noun/num_words, verb/num_words, adj/num_words], 1)

        return [np.log(num_words), num_unique_words/num_words, sentiment, noun/num_words, verb/num_words, adj/num_words, 1]
    """

    def featurize(self, lyrics):
        words = lyrics.split()
        if len(words) > FEATURE_WORDS:
            words = words[:FEATURE_WORDS]
        elif len(words) < FEATURE_WORDS:
            words.extend([PAD] * (FEATURE_WORDS-len(words)))
        sliced = " ".join(words)
        tokens = self.embeddings(sliced)
        vectors = []
        for vec in [tok.vector for tok in tokens]:
            vectors.extend(vec)
        print(len(vectors))
        #print(vectors[-5:])
        return vectors

    def preprocess_lyrics(self, lyrics):
        stop_words = [".", ",", "\n", "a", "the", "is", "i", "am", "are"]
        lyrics = lyrics.lower()
        for word in stop_words:
            lyrics = lyrics.replace(word, "")
        return lyrics
        """ More technically effective but worse results
        processed_lyrics = ""
        for word in lyrics.lower().replace(".", "").replace(",", "").split():
            if word not in STOPWORDS:
                processed_lyrics += (word + " ")
        return processed_lyrics[0:-1]"""

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
            print(counter, len(x_gen), len(y_gen))
            yield np.array(x_gen), np.array(y_gen)
            counter += num_sequences_per_batch

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
        self.weights = [[1, 1, 1, 1, 1, 1, self.bias] for _ in range(len(class_labels))]
        self.learning_rate = 0.012

    def train(self, songs):
        # Updates the classifier against given input examples
        for song in songs:
            features = self.featurize(song.lyrics)
            n = [np.dot(weight, features) for weight in self.weights]
            prob = softmax(n)
            #print(prob)
            # Calculate gradients for each weight
            idx = self.class_labels.index(song.artist)
            grads = (-np.log(prob[idx] + 0.00001)) * np.array(features)

            # Update weights
            self.weights[idx] = self.weights[idx] + (self.learning_rate * grads)
            local_bias = self.weights[idx][-1]
            self.weights = [np.append(weight[:-1], local_bias) for weight in self.weights]
        print(self.weights)

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
        self.nn.add(Dense(1000, input_shape=(FEATURE_LENGTH,), activation='relu'))
        self.nn.add(Dense(len(self.class_labels), activation='softmax'))
        self.nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, songs):
        steps_per_epoch = len(songs) // self.num_sequences_per_batch
        X = [song.lyrics for song in songs]
        y = [song.artist for song in songs]
        data_gen = self.data_generator(X, y, self.num_sequences_per_batch)
        self.nn.fit(x=data_gen, epochs=1, steps_per_epoch=steps_per_epoch)

    def classify(self, song_lyrics):
        features = self.featurize(song_lyrics)
        return self.nn.predict([features])


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
