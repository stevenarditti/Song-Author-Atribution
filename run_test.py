import artist_classifier
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin', "Aerosmith", "Frank Sinatra", "Kanye West", "Eminem", "Red Hot Chili Peppers", "Queen", "Billy Joel", "Madonna", "Drake"]


def plots_data(data):

    for data_set in data:
        plt.boxplot(data_set)

    plt.show()
    plt.savefig('box_accuracy.png')

def metrics(model, songs):
    predicted = []
    true_labels = []
    for song in songs:
        predicted_artist = model.classify(artist_classifier.preprocess_lyrics(song.lyrics))
        predicted.append(predicted_artist)
        true_labels.append(song.artist)
    print(f"{model}:")
    print(classification_report(true_labels, predicted, digits=3, zero_division=0))
    return classification_report(true_labels, predicted, digits=3, zero_division=0, output_dict=True)


def main():
    songs = artist_classifier.load_data()
    random.shuffle(songs)

    data = [[], [], []]

    # Used to clean the input songs from genius
    cleaned_lyrics = [" ".join(artist_classifier.lines_from_song(song.lyrics)) for song in songs]
    for idx, song in enumerate(songs):
        song.lyrics = artist_classifier.preprocess_lyrics(cleaned_lyrics[idx])

    train_data, test_data = train_test_split(songs)


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


    bow_report = metrics(bow_classifier, test_data)
    data[0].append(bow_report['accuracy'])
    
    lr_report = metrics(lr_classifier, test_data)
    data[1].append(lr_report['accuracy'])

    nn_report = metrics(ffnn_classifier, test_data)
    data[2].append(nn_report['accuracy'])



if __name__ == '__main__':
    main()
