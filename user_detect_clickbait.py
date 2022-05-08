import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm

from sklearn import metrics


# Used to increase parallel computing on CPU
# pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()



# trainClassifiers() runs a compiled .csv file through all four classifiers of over 30,000 data points.
def trainClassifiers():

    print("\nTraining classifiers...", end="")

     # Defining Constants
    clickbait_title_column = "headline"
    is_clickbait_column = "clickbait"

    # Pandas dataframe df
    df = pd.read_csv("input_data/clickbait_compilation.csv")


    y = df[is_clickbait_column]
    X = df[clickbait_title_column].str.lower() # All titles are in lowercase


    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(X)


    # MultinomialNB
    multiNB_clf = MultinomialNB(alpha=0.00001)
    multiNB_clf.fit(vectors, y)

    # SGD
    SGD_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    SGD_clf.fit(vectors, y)

    # Perceptron
    perceptron_clf = Perceptron(tol=1e-3, random_state=0)
    perceptron_clf.fit(vectors, y)

    # SVM
    SVM_clf = svm.SVC(gamma=3, kernel='sigmoid')
    SVM_clf.fit(vectors, y)

    print("done")

    return vectorizer, multiNB_clf, SGD_clf, perceptron_clf, SVM_clf


# predictWithClassifiers() takes the user's entry and predicts clickbait status on all four classifiers
def predictWithClassifiers(user_clickbait_title, vectorizer, multiNB_pred, SGD_pred, perceptron_pred, SVM_pred):
        
        print("\n-----------------------------------------------------------")
        print("STATISTICS:")

        # Vectorize the title to pass into the prediction of the classifier
        vector_user_title_test = vectorizer.transform(user_clickbait_title)

        # Multinomial Prediction
        multiNB_pred = multiNB_clf.predict(vector_user_title_test)
        print("Multinomial Naive Bayes Prediction:\t\t", multiNB_pred)


        # Stochastic Gradient Descent Prediction
        SGD_pred = SGD_clf.predict(vector_user_title_test)
        print("Stochastic Gradient Descent Prediction:\t\t", SGD_pred)


        # Perceptron Prediction
        perceptron_pred = perceptron_clf.predict(vector_user_title_test)
        print("Perceptron Prediction:\t\t\t\t", perceptron_pred)
        

        # SVM prediction
        SVM_pred = SVM_clf.predict(vector_user_title_test)
        print("Support Vector Machine Prediction:\t\t", SVM_pred)

        return multiNB_pred, SGD_pred, perceptron_pred, SVM_pred


# classifierClickbaitStatus()) finds the clickbait status of all four classifiers by averaging their predictions
# Prediction values are [0] for a non-clickbait title and [1] for a clickbait title
def classifierClickbaitRatio(multiNB_pred, SGD_pred, perceptron_pred, SVM_pred):
    classifier_clickbait_ratio = (multiNB_pred+SGD_pred+perceptron_pred+SVM_pred)/ num_of_classifiers

    print("\nTotal Classification Clickbait Status:\t\t\t", classifier_clickbait_ratio)
    print("\n-----------------------------------------------------------")

    return classifier_clickbait_ratio


# determineClickbait() determines if the title is clickbait based on the four classifier predictions
def determineClickbait(total_classifier_accuracy, prediction_threshold):
    # If the classifier accuracy is above the threshold, title is clickbait
    # Else, it's below the threshold, so title is not clickbait
    
    if (total_classifier_accuracy >= prediction_threshold):
        print("\nCLICKBAIT!")
    else:
        print("\nNOT CLICKBAIT")

    print("\n------------")



if __name__ == '__main__':

    # Constants
    num_of_classifiers = 4
    prediction_threshold = 0.75

    # train Classifiers on 30,000 data points
    vectorizer, multiNB_clf, SGD_clf, perceptron_clf, SVM_clf = trainClassifiers()

    # Get user clickbait title
    user_clickbait_title = input("\n\nEnter a title to see if it's clickbait (-1 to quit):\n").lower()

    # Allows the user to enter and determine many clickbait titles per program execution
    # Quits program when a string "-1" is entered
    while (user_clickbait_title != "-1"):

        # Turn string title into list iterable for vectorization
        user_clickbait_title = [user_clickbait_title]
        
        # Predicting the clickbait classification on all classifiers
        multiNB_pred, SGD_pred, perceptron_pred, SVM_pred = predictWithClassifiers(user_clickbait_title, vectorizer, multiNB_clf, SGD_clf, perceptron_clf, SVM_clf)

        # Total mean classifier accuracy to determine if the title is clickbait
        classifier_clickbait_ratio = classifierClickbaitRatio(multiNB_pred, SGD_pred, perceptron_pred, SVM_pred)

        # Determing that clickbait status of the user's title
        determineClickbait(classifier_clickbait_ratio, prediction_threshold)

        # Get user clickbait title for another entry
        user_clickbait_title = input("\nEnter a title to see if it's clickbait (-1 to quit):\n")
        


    print("\nBye now!\n")