# -*- coding: utf-8 -*-
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# reading the data from broad_category_data.obj
with open ('input/broad_category_data.obj', 'rb') as fp:
    broad_category_data = pickle.load(fp)

# Array of [sequence]
pmids = []
docs = []
broad_categories = []
str = ""
for element in broad_category_data:
    pmids = []
    sequence = [element[1], element[2]]
    str = (u''.join(sequence))
    docs.append(str)
    broad_categories.append(element[3])

sklearn_tfidf = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True)
tfidf_representation = sklearn_tfidf.fit_transform(docs)

print tfidf_representation.shape

# with open('output/tfidf_representation.obj', 'wb') as fp:
#     pickle.dump(tfidf_representation, fp)
#
# with open('output/tfidf_categories.obj', 'wb') as fp:
#     pickle.dump(broad_categories, fp)


# split example data into training set(60%), cross-validation set(20%), and test set(20%)
tfidf_vecs = tfidf_representation.toarray()
broad_category = np.array(broad_categories)
sss1 = StratifiedShuffleSplit(test_size=0.4)
for train_index, intermediate_test_index in sss1.split(tfidf_vecs, broad_category):
    # Array of [pmid, cleaned_sequence]
    doc_train, intermediate_doc_test = tfidf_vecs[train_index], tfidf_vecs[intermediate_test_index]
    category_train, intermediate_category_test = broad_category[train_index], broad_category[intermediate_test_index]

sss2 = StratifiedShuffleSplit(test_size=0.5)
for cv_index, test_index in sss2.split(intermediate_doc_test, intermediate_category_test):
    # Array of [pmid, cleaned_sequence]
    doc_cv, doc_test = intermediate_doc_test[cv_index], intermediate_doc_test[test_index]
    category_cv, category_test = intermediate_category_test[cv_index], intermediate_category_test[test_index]

with open('output/tfidf_train.obj', 'wb') as fp:
     pickle.dump(doc_train, fp)

with open('output/tfidf_cv.obj', 'wb') as fp:
     pickle.dump(doc_cv, fp)

with open('output/tfidf_test.obj', 'wb') as fp:
     pickle.dump(doc_test, fp)

with open('output/category_train.obj', 'wb') as fp:
     pickle.dump(category_train, fp)

with open('output/category_cv.obj', 'wb') as fp:
     pickle.dump(category_cv, fp)

with open('output/category_test.obj', 'wb') as fp:
     pickle.dump(category_test, fp)
