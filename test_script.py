import artist_classifier
import api_data_retriever
import random

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin']


def getAccuracy(model, songs):
    print(f"\nAccuracy for {model}:")
    correct = incorrect = 0
    song_count = {artist: 0 for artist in ARTISTS}
    for song in songs:
        predicted_artist = model.classify(artist_classifier.preprocess_lyrics(song.lyrics))
        if song.artist == predicted_artist:
            correct += 1
        else:
            incorrect += 1
        song_count[predicted_artist] += 1

    print("Correct:", correct, "/", len(songs))
    for artist in ARTISTS:
        print(artist + ": " + str(song_count[artist]))
    print("\n")


def main():
    songs = artist_classifier.load_data()
    random.shuffle(songs)

    # Used to clean the input songs from genius
    cleaned_lyrics = [" ".join(artist_classifier.lines_from_song(song.lyrics)) for song in songs]
    for idx, song in enumerate(songs):
        song.lyrics = artist_classifier.preprocess_lyrics(cleaned_lyrics[idx])

    bow_classifier = artist_classifier.Bag_of_Words_Artist_Classifier("Test Bag-of-Words", ARTISTS)
    bow_classifier.train(songs)
    getAccuracy(bow_classifier, songs)

    lr_classifier = artist_classifier.Logistic_Regression_Artist_Classifier("Test LogRes", ARTISTS)
    lr_classifier.train(songs)
    getAccuracy(lr_classifier, songs)

    return

    ffnn_classifier = artist_classifier.Feed_Forward_Neural_Net_Artist_Classifier("Test FFNN", ARTISTS)
    ffnn_classifier.train(songs)
    getAccuracy(ffnn_classifier, songs)


if __name__ == '__main__':
    main()
