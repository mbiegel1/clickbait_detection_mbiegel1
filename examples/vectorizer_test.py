from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# Used to increase parallel computing on CPU
# pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()

# Create our vectorizer
vectorizer = CountVectorizer()

# All data
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'))

# Get the training vectors
vectors = vectorizer.fit_transform(newsgroups_train.data)

# Build the classifier
clf = MultinomialNB(alpha=.01)

print (type(newsgroups_train.target))
#print("Vectors:", vectors)

#  Train the classifier
clf.fit(vectors, newsgroups_train.target)
print("Size of vectors:", vectors.shape)
print("Size of newsgroups_train.target:", newsgroups_train.target.shape)

# Get the test vectors
vectors_test = vectorizer.transform(newsgroups_test.data)
print("Size of vectors_test:", vectors_test.shape)
print("Datatype passed into predict:", type(vectors_test))

# Predict and score the vectors
pred = clf.predict(vectors_test)
acc_score = metrics.accuracy_score(newsgroups_test.target, pred)
f1_score = metrics.f1_score(newsgroups_test.target, pred, average='macro')

print('Total accuracy classification score: {}'.format(acc_score))
print('Total F1 classification score: {}'.format(f1_score))

#print("\n\n\n", vectors_test[0])

# print("\n", newsgroups_test.data[0])

# print("\n", newsgroups_test.target[0])
# print(type(newsgroups_test.target))

# print("\n",pred)
# print(type(pred))