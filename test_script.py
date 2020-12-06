import artist_classifier
import api_data_retriever
import random

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin']


def getAccuracy(model, songs):
    correct = incorrect = 0
    song_count = {'Kendrick Lamar': 0, 'The Beatles': 0, 'Led Zeppelin': 0}
    for song in songs:
        predicted_artist = model.classify(model.preprocess_lyrics(song.lyrics))
        if song.artist == predicted_artist:
            correct += 1
        else:
            incorrect += 1
        song_count[predicted_artist] += 1

    print("Correct:", correct, "/", len(songs))
    for artist in ARTISTS:
        print(artist + ": " + str(song_count[artist]))


def main():
    songs = artist_classifier.load_data()
    random.shuffle(songs)

    #bow_classifier = artist_classifier.Bag_of_Words_Artist_Classifier("Test Bag-of-Words", ARTISTS)
    #bow_classifier.train(songs)
    #getAccuracy(bow_classifier, songs)

    #lr_classifier = artist_classifier.Logistic_Regression_Artist_Classifier("Test LogRes", ARTISTS)
    #lr_classifier.train(songs)
    #getAccuracy(lr_classifier, songs)

    ffnn_classifier = artist_classifier.Feed_Forward_Neural_Net_Artist_Classifier("Test FFNN", ARTISTS)
    ffnn_classifier.train(songs)
    getAccuracy(ffnn_classifier, songs)


if __name__ == '__main__':
    main()
