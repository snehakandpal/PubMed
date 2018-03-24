from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import pickle
import numpy as np

# reading the data from train_vector.obj, cv_vector.obj and test.obj
with open ('output/train_vector.obj', 'rb') as fp:
    train_vector = pickle.load(fp)

with open ('output/cv_vector.obj', 'rb') as fp:
    cv_vector = pickle.load(fp)

with open ('output/test_vector.obj', 'rb') as fp:
    test_vector = pickle.load(fp)

#labels
with open ('output/category_train.obj', 'rb') as fp:
    category_train = pickle.load(fp)

with open ('output/category_cv.obj', 'rb') as fp:
    category_cv = pickle.load(fp)

with open ('output/category_test.obj', 'rb') as fp:
    category_test = pickle.load(fp)

#SVM classifier
classifier = OneVsRestClassifier(SVC(kernel='sigmoid', coef0=0.9))
# classifier = SVC(kernel='rbf', C=1, gamma=1e-05, coef0=0)
classifier.fit(train_vector, category_train)

# parameters = {'kernel':('rbf', 'sigmoid'), 'C':[1, 1000], 'gamma':[0.00001, 20000], 'coef0':[0, 100]}
# grid_search = GridSearchCV(svm.SVC(), parameters)
# grid_search.fit(cv_vector, category_cv)
# print grid_search.best_params_

#cv set
print classifier.score(cv_vector, category_cv)

#check accuracy
# print classifier.score(test_vector, category_test)

#to predict label
#predicted_test = classifier.predict(test_vector)
