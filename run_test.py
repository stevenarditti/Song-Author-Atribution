import artist_classifier
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin', "Aerosmith", "Frank Sinatra", "Kanye West", "Eminem", "Red Hot Chili Peppers", "Queen", "Billy Joel", "Madonna", "Drake"]

def print_report(model, songs):
    """ Prints metric report of given model
        Parameters:
            model: model to calculate metrics
            songs: list of songs
        Returns:
            void
    """
    predicted = []
    true_labels = []
    for song in songs:
        predicted_artist = model.classify(artist_classifier.preprocess_lyrics(song.lyrics))
        predicted.append(predicted_artist)
        true_labels.append(song.artist)
    print(f"{model}:")
    print(classification_report(true_labels, predicted, digits=3, zero_division=0))

def plot_song_len(songs):
    """ Generates histogram of word count per song
        Parameters:
            songs: list of songs
        Returns:
            void
    """
    lengths = []
    sub_200 = 0
    sub_300 = 0
    total_words = 0
    for song in songs:
        word_count = len(song.lyrics.split())
        lengths.append(word_count)
        if word_count < 200:
            sub_200 += 1
        if word_count < 300:
            sub_300 += 1
        total_words += word_count
    
    print(f"{(sub_200 * 100) // len(songs)}% of songs below 200")
    print(f"{(sub_300 * 100) // len(songs)}% of songs below 300")
    print(f"{total_words} total words across all songs")

    plt.hist(lengths, bins='auto')
    plt.xlabel("word count")
    plt.ylabel("# songs")
    plt.title("Words count vs number of songs")
    plt.show()


def main():
    songs = artist_classifier.load_data()
    random.shuffle(songs)

    # Used to clean the input songs from genius
    cleaned_lyrics = [" ".join(artist_classifier.lines_from_song(song.lyrics)) for song in songs]
    for idx, song in enumerate(songs):
        song.lyrics = artist_classifier.preprocess_lyrics(cleaned_lyrics[idx])
    plot_song_len(songs)

    train_data, test_data = train_test_split(songs)

    # train models
    print("Training bag of words classifier...")
    bow_classifier = artist_classifier.Bag_of_Words_Artist_Classifier("Test Bag-of-Words", ARTISTS)
    bow_classifier.train(train_data)
    print("BOW done.")
    

    print("Training logistic regression classifier...")
    lr_classifier = artist_classifier.Logistic_Regression_Artist_Classifier("Test LogRes", ARTISTS)
    lr_classifier.train(train_data)
    print("Logreg done.\n")

    print("Training neural network classifier...")
    ffnn_classifier = artist_classifier.Feed_Forward_Neural_Net_Artist_Classifier("Test FFNN", ARTISTS)
    ffnn_classifier.train(train_data)
    print("NN done.")


    # print reports
    print_report(bow_classifier, test_data)
    print_report(lr_classifier, test_data)
    print_report(ffnn_classifier, test_data)


if __name__ == '__main__':
    main()
