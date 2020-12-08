import artist_classifier
import random
from sklearn.model_selection import train_test_split

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin', "Aerosmith", "Frank Sinatra", "Kanye West", "Eminem", "Red Hot Chili Peppers", "Queen", "Billy Joel", "Madonna", "Drake"]


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

    train_data, test_data = train_test_split(songs)

    bow_classifier = artist_classifier.Bag_of_Words_Artist_Classifier("Test Bag-of-Words", ARTISTS)
    bow_classifier.train(train_data)
    getAccuracy(bow_classifier, test_data)

    lr_classifier = artist_classifier.Logistic_Regression_Artist_Classifier("Test LogRes", ARTISTS)
    lr_classifier.train(train_data)
    getAccuracy(lr_classifier, test_data)

    ffnn_classifier = artist_classifier.Feed_Forward_Neural_Net_Artist_Classifier("Test FFNN", ARTISTS)
    ffnn_classifier.train(train_data)
    getAccuracy(ffnn_classifier, test_data)


if __name__ == '__main__':
    main()
