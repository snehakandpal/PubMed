import pickle
import string
import numpy as np
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.utils import to_unicode
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import utils

# reading the data from broad_category_data.obj
with open ('input/broad_category_data.obj', 'rb') as fp:
    broad_category_data = pickle.load(fp)


# Array of [pmid, sequence]
doc = []
broad_category = []
str = ""
for element in broad_category_data:
    sequence = [element[1], element[2]]
    str = (u''.join(sequence))
    doc.append([element[0], str])
    broad_category.append(element[3])

#text preprocessing
def doc_preprocess(doc):
    temp_doc = []
    for s in doc:
        corpus  = s[1].translate(string.punctuation)
        corpus = corpus.lower()
        corpus = corpus.encode('ascii','ignore')
        corpus = ' '.join([word for word in corpus.split() if word not in stopwords.words("english")])
        temp_doc.append([s[0], corpus])
    return temp_doc

# Array of [pmid, cleaned_sequence]
clean_doc = doc_preprocess(doc) #####also convert category to ascii

# split example data into training set(60%), cross-validation set(20%), and test set(20%)
clean_doc = np.array(clean_doc)
broad_category = np.array(broad_category)
sss1 = StratifiedShuffleSplit(test_size=0.4)
for train_index, intermediate_test_index in sss1.split(clean_doc, broad_category):
    # Array of [pmid, cleaned_sequence]
    doc_train, intermediate_doc_test = clean_doc[train_index], clean_doc[intermediate_test_index]
    category_train, intermediate_category_test = broad_category[train_index], broad_category[intermediate_test_index]

sss2 = StratifiedShuffleSplit(test_size=0.5)
for cv_index, test_index in sss2.split(intermediate_doc_test, intermediate_category_test):
    # Array of [pmid, cleaned_sequence]
    doc_cv, doc_test = intermediate_doc_test[cv_index], intermediate_doc_test[test_index]
    category_cv, category_test = intermediate_category_test[cv_index], intermediate_category_test[test_index]


# At this point we have following Array of [pmid, cleaned_sequence]
# doc_train, doc_cv, doc_test

#tag doc
def tag_doc(document):
    tagged = []
    for index, text in enumerate(document):
        tag = '%s_%s_%s'%(text[0], 'uuid', index)
        tagged.append(TaggedDocument(to_unicode(text[1]), [tag]))
    return tagged

tagged_doc = tag_doc(clean_doc)
tagged_doc_train = tag_doc(doc_train)
tagged_doc_cv = tag_doc(doc_cv)
tagged_doc_test = tag_doc(doc_test)

# At this point we have following Array of tagged_document
# tagged_doc, tagged_doc_train, tagged_doc_cv, tagged_doc_test

# #doc2vec
model = Doc2Vec(dm=0, vector_size=10, min_count=5, epochs=10)
# #min_count, size and dm(0->distribuuted memory, 1->distributed bag of words) with cv
# #build vocab
model.build_vocab(tagged_doc)
print ("Corpus count: ", model.corpus_count)
print ("Tagged docs count: ", len(tagged_doc))
model.train(utils.shuffle(tagged_doc), total_examples=model.corpus_count, epochs=model.epochs)

#save the model
model.save('output/trained_model')

# get document vector
def get_vector(model, doc_type):
    j = 0
    k = 0
    if doc_type == 'train':
        j = 0
        k = len(tagged_doc_train)
    if doc_type == 'cv':
        j = len(tagged_doc_train)
        k = j + len(tagged_doc_cv)
    if doc_type == 'test':
        j = len(tagged_doc_train) + len(tagged_doc_cv)
        k = j + len(tagged_doc_test)

    vector = []
    for i in range(j , k):
        vector.append(model.docvecs[i])

    return np.array(vector)

# trained_model = Doc2Vec.load('trained_model')

print ("Doc vectors trained: ", len(model.docvecs))

train_vector = get_vector(model, 'train')
print "train_vector"
print train_vector.shape

cv_vector = get_vector(model, 'cv')
print "cv_vector"
print cv_vector.shape

test_vector = get_vector(model, 'test')
print "test_vector"
print test_vector.shape

# sims = model.docvecs.most_similar(test_vector[ [0,1], ])
# print "Sims:"
# print sims

with open('output/train_vector.obj', 'wb') as fp:
    pickle.dump(train_vector, fp)

with open('output/cv_vector.obj', 'wb') as fp:
    pickle.dump(cv_vector, fp)

with open('output/test_vector.obj', 'wb') as fp:
    pickle.dump(test_vector, fp)

with open('output/category_train.obj', 'wb') as fp:
    pickle.dump(category_train, fp)

with open('output/category_cv.obj', 'wb') as fp:
    pickle.dump(category_cv, fp)

with open('output/category_test.obj', 'wb') as fp:
    pickle.dump(category_test, fp)
