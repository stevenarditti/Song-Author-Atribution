import artist_classifier
import api_data_retriever

ARTISTS = ['Kendrick Lamar', 'The Beatles', 'Led Zeppelin']

def main():
    songs = artist_classifier.load_data()

    #bow_classifier = artist_classifier.Bag_of_Words_Artist_Classifier("Test Bag-of-Words", ARTISTS)
    #bow_classifier.train(songs)

    lr_classifier = artist_classifier.Logistic_Regression_Artist_Classifier("Test LogRes", ARTISTS)
    lr_classifier.train(songs)


if __name__ == '__main__':
    main()
