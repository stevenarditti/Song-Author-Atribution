import artist_classifier
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin', "Aerosmith", "Frank Sinatra", "Kanye West", "Eminem", "Red Hot Chili Peppers", "Queen", "Billy Joel", "Madonna", "Drake"]


def print_metrics(model, songs):
    print(f"\nAccuracy for {model}:")
    correct = incorrect = 0
    song_count = {artist: 0 for artist in ARTISTS}
    predicted = []
    true_labels = []
    for song in songs:
        predicted_artist = model.classify(artist_classifier.preprocess_lyrics(song.lyrics))
        predicted.append(predicted_artist)
        true_labels.append(song.artist)

        if song.artist == predicted_artist:
            correct += 1
        else:
            incorrect += 1
        song_count[predicted_artist] += 1

    accuracy = artist_classifier.accuracy(predicted, true_labels)
    print(f"Accuracy: {accuracy}", )
    print(f"F score: {artist_classifier.f1_score(predicted, true_labels)}")
    for artist in ARTISTS:
        print(artist + ": " + str(song_count[artist]))
    print(f"{correct} / {len(songs)}")

def metrics(model, songs):
    predicted = []
    true_labels = []
    for song in songs:
        predicted_artist = model.classify(artist_classifier.preprocess_lyrics(song.lyrics))
        predicted.append(predicted_artist)
        true_labels.append(song.artist)
    print(f"{model}:")
    print(classification_report(true_labels, predicted, digits=3, zero_division=0))

def main():
    songs = artist_classifier.load_data()
    random.shuffle(songs)

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

    metrics(bow_classifier, test_data)
    
    metrics(lr_classifier, test_data)

    metrics(ffnn_classifier, test_data)


if __name__ == '__main__':
    main()
