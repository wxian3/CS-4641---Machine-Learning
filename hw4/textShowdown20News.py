'''
    AUTHOR Wenqi Xian
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from pprint import pprint
import sklearn.metrics as smet
import time

# Load Data set
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True, random_state = 42)
newsgroups_test  = fetch_20newsgroups(subset='test' , shuffle = True, random_state = 42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
X_test_counts  = count_vect.transform(newsgroups_test.data)

tfidf_transformer = TfidfTransformer(norm = 'l2', sublinear_tf = True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train Multinomial Naives Bayes
NB = MultinomialNB()
begin = time.clock()
NB.fit(X_train_tfidf, newsgroups_train.target)
end = time.clock()
NB_train_time = end - begin

# Predict Multinomial Naives Bayes
NB_train_predicted = NB.predict(X_train_tfidf)
NB_test_predicted  = NB.predict(X_test_tfidf)

# Calculate performance statistics
NB_train_accuracy = smet.accuracy_score(newsgroups_train.target, NB_train_predicted)
NB_test_accuracy  = smet.accuracy_score(newsgroups_test.target,  NB_test_predicted)

NB_train_precision = smet.precision_score(newsgroups_train.target, NB_train_predicted, average = 'macro')
NB_test_precision  = smet.precision_score(newsgroups_test.target,  NB_test_predicted, average = 'macro')

NB_train_recall = smet.recall_score(newsgroups_train.target, NB_train_predicted, average = 'macro')
NB_test_recall  = smet.recall_score(newsgroups_test.target,  NB_test_predicted, average = 'macro')

# Train SVM with Cosine Kernel
SVM = svm.SVC(kernel = smet.pairwise.linear_kernel, probability = True)
begin = time.clock()
SVM.fit(X_train_tfidf, newsgroups_train.target)
end = time.clock()
SVM_train_time = end - begin

# Predict SVM with Cosine Kernel
SVM_train_predicted = SVM.predict(X_train_tfidf)
SVM_test_predicted  = SVM.predict(X_test_tfidf)

# Calculate performance statistics
SVM_train_accuracy = smet.accuracy_score(newsgroups_train.target, SVM_train_predicted)
SVM_test_accuracy  = smet.accuracy_score(newsgroups_test.target,  SVM_test_predicted)

SVM_train_precision = smet.precision_score(newsgroups_train.target, SVM_train_predicted, average = 'macro')
SVM_test_precision  = smet.precision_score(newsgroups_test.target,  SVM_test_predicted, average = 'macro')

SVM_train_recall = smet.recall_score(newsgroups_train.target, SVM_train_predicted, average = 'macro')
SVM_test_recall  = smet.recall_score(newsgroups_test.target,  SVM_test_predicted, average = 'macro')


print("+----------+-------------------------+-------------------------+")
print("|Classifier|       Naives Bayes      |  SVM with Cosine Kernel |")
print("+----------+-------------------------+-------------------------+")
print("|          |    Train   |    Test    |    Train   |    Test    |")
print("+----------+------------+------------+------------+------------+")
print("| Accuracy | %2.2f %%    | %2.2f %%    | %2.2f %%    | %2.2f %%    |" % (NB_train_accuracy *100, NB_test_accuracy  *100, SVM_train_accuracy *100, SVM_test_accuracy  *100))
print("+----------+------------+------------+------------+------------+")
print("| Precision| %2.2f %%    | %2.2f %%    | %2.2f %%    | %2.2f %%    |" % (NB_train_precision *100, NB_test_precision  *100, SVM_train_precision *100, SVM_test_precision  *100))
print("+----------+------------+------------+------------+------------+")
print("| Recall   | %2.2f %%    | %2.2f %%    | %2.2f %%    | %2.2f %%    |" % (NB_train_recall *100, NB_test_recall  *100, SVM_train_recall *100, SVM_test_recall  *100))
print("+----------+------------+------------+------------+------------+")
print("|Train Time|        %2.2f s           |        %2.2f s          |" % (NB_train_time, SVM_train_time))
print("+----------+-------------------------+-------------------------+")
