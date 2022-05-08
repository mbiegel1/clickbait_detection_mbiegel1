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



if __name__ == '__main__':


    
    print("\n\n-------------------------CURRENT DATA-----------------------------------------\n")
    # Reading data into pandas dataframe

    # Defining Constants
    clickbait_title_column = "headline"
    is_clickbait_column = "clickbait"

    # Pandas dataframe df
    df = pd.read_csv("input_data/clickbait_consensus.csv")

    y = df[is_clickbait_column]
    X = df[clickbait_title_column]



    print("\n\n-------------------------VECTORIZING-----------------------------------------\n")
    # Vectorizing data in order for classifiers to use them
    # This is required because classifiers need to numerical data, 
    #   so the titles (wihch are strings) need to be transformed numerically
    
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(X)

    
    
    print("\n\n-------------------------TEST DATA-----------------------------------------\n")
    # Asking user what data to test classifiers on

    which_data = input("\nWould you like to test on training data (enter 0) or pre-determined external data (enter 1)? ")

    if (which_data == "0"):
        print("Training data chosen")
        test_multi_titles = df[clickbait_title_column]
        test_multi_titles_nparray = df[is_clickbait_column].to_numpy()

        vectors_test = vectorizer.transform(test_multi_titles)
    
    else:
        print("External data chosen")
        
        test_df = pd.read_csv("input_data/clickbait_ratio_flattened.csv")

        test_multi_titles = test_df[clickbait_title_column]
        test_multi_titles_nparray = test_df[is_clickbait_column].to_numpy()

        vectors_test = vectorizer.transform(test_multi_titles)



    print("\n\n-------------------------MULTINOMIALNB PREDICTING-----------------------------------------\n")
    # Multinomial Naive Bayes Classification

    multiNB_clf = MultinomialNB(alpha=0.00001)
    multiNB_clf.fit(vectors, y)


    multiNB_pred = multiNB_clf.predict(vectors_test)

    print("Prediction:", multiNB_pred)
    print()

    acc_score_multiNB = metrics.accuracy_score(test_multi_titles_nparray, multiNB_pred)
    f1_score_multNB = metrics.f1_score(test_multi_titles_nparray, multiNB_pred, average='macro')

    print('Total accuracy classification score: {}'.format(acc_score_multiNB))
    print('Total F1 classification score: {}'.format(f1_score_multNB))



    print("\n\n-------------------------STOCHSTIC GRADIENT DESCENT (SGD) PREDICTING---------------------------------------\n")
    # Stochastic Gradient Descent Classification

    SGD_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    SGD_clf.fit(vectors, y)

    SGD_pred = SGD_clf.predict(vectors_test)

    print("Prediction:", SGD_pred)
    print()

    acc_score_SGD = metrics.accuracy_score(test_multi_titles_nparray, SGD_pred)
    f1_score_SGD = metrics.f1_score(test_multi_titles_nparray, SGD_pred, average='macro')

    print('Total accuracy classification score: {}'.format(acc_score_SGD))
    print('Total F1 classification score: {}'.format(f1_score_SGD))
    print()



    print("\n\n-------------------------PERCEPTRON PREDICTING---------------------------------------\n")
    # Perceptron Classification

    perceptron_clf = Perceptron(tol=1e-3, random_state=0)
    perceptron_clf.fit(vectors, y)

    perceptron_pred = perceptron_clf.predict(vectors_test)

    print("Prediction:", perceptron_pred)
    print()

    acc_score_perceptron = metrics.accuracy_score(test_multi_titles_nparray, perceptron_pred)
    f1_score_perceptron = metrics.f1_score(test_multi_titles_nparray, perceptron_pred, average='macro')

    print('Total accuracy classification score: {}'.format(acc_score_perceptron))
    print('Total F1 classification score: {}'.format(f1_score_perceptron))
    


    print("\n\n-------------------------SVM PREDICTING---------------------------------------\n")
    # Support Vector Machine Classification
    
    SVM_clf = svm.SVC(gamma=3, kernel='sigmoid')
    SVM_clf.fit(vectors, y)

    SVM_pred = SVM_clf.predict(vectors_test)
    
    print("Prediction:", SVM_pred)
    print()

    acc_score_SVM = metrics.accuracy_score(test_multi_titles_nparray, SVM_pred)
    f1_score_SVM = metrics.f1_score(test_multi_titles_nparray, SVM_pred, average='macro')

    print('Total accuracy classification score: {}'.format(acc_score_SVM))
    print('Total F1 classification score: {}'.format(f1_score_SVM))


    print("\n")