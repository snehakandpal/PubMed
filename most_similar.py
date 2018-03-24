import pickle
import numpy as np
# from numpy import linalg as LA
from scipy.spatial import distance

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

# train_vector = np.concatenate((train_vector, cv_vector), axis=0)
# category_train = [x.encode('UTF8') for x in category_train]
# category_cv = [x.encode('UTF8') for x in category_cv]
# category_train = category_train + category_cv
category_train = [x.encode('UTF8') for x in category_train]

# print train_vector
# print len(category_train)

com = noncom = oth = pl = an = nutri = rmnc = inj = phy = chem = j = 0
communicable = []
noncommunicable = []
others = []
plant = []
animal = []
nutrition = []
rmnch = []
injuries = []
physical = []
chemical = []

for i in range(len(train_vector)):
    if category_train[i].lower() ==  "non-communicable disease":
        noncom +=1
        noncommunicable.append([train_vector[i]])
    elif category_train[i].lower() == "communicable disease":
        com +=1
        communicable.append([train_vector[i]])
    elif category_train[i].lower() == "others":
        oth +=1
        others.append([train_vector[i]])
    elif category_train[i].lower() == "plant research":
        pl +=1
        plant.append([train_vector[i]])
    elif category_train[i].lower() == "animal research":
        an +=1
        animal.append([train_vector[i]])
    elif category_train[i].lower() == "nutrition":
        nutri +=1
        nutrition.append([train_vector[i]])
    elif category_train[i].lower() == "rmnch":
        rmnc +=1
        rmnch.append([train_vector[i]])
    elif category_train[i].lower() == "injuries":
        inj +=1
        injuries.append([train_vector[i]])
    elif category_train[i].lower() == "physical sciences":
        phy +=1
        physical.append([train_vector[i]])
    elif category_train[i].lower() == "chemical sciences":
        chem +=1
        chemical.append([train_vector[i]])

print com, noncom, oth, pl, an, nutri, rmnc, inj, phy, chem
print len(communicable), len(noncommunicable), len(others), len(plant), len(animal), len(nutrition), len(rmnch), len(injuries), len(physical), len(chemical)

average = []
average.append(np.mean(communicable, axis=0))
average.append(np.mean(noncommunicable, axis=0))
average.append(np.mean(others, axis=0))
average.append(np.mean(plant, axis=0))
average.append(np.mean(animal, axis=0))
average.append(np.mean(nutrition, axis=0))
average.append(np.mean(rmnch, axis=0))
average.append(np.mean(injuries, axis=0))
average.append(np.mean(physical, axis=0))
average.append(np.mean(chemical, axis=0))

print average

def compare_vector(test):
    min = 99999
    for i in range(10):
        # distance = np.linalg.norm(average[i]-test)
        # dist = distance.euclidean(average[i], test)
        dist = distance.cosine(average[i], test)

        if(dist < min):
            min = dist
            index = i
    return index

categories = ["communicable", "non-communicable", "others", "plant research", "animal research",
              "nutrition", "rmnch", "injuries", "physical sciences", "chemical sciences"]

predicted_cat = []

category_counter = {}
def find_category():
    for i in range(len(train_vector)):
        cat_index = compare_vector(train_vector[i])
        predicted_cat.append(categories[cat_index])
        if categories[cat_index] in category_counter:
            category_counter[categories[cat_index]] = category_counter[str(categories[cat_index])] + 1
        else:
            category_counter[categories[cat_index]] = 1

find_category()

print category_counter

# print category_test
# print predicted_cat

def accuracy():
    for i in range(len(train_vector)):
        true = 0
        if(predicted_cat[i] == category_train[i].lower()):
            true += 1
        # else:
        #     print "-----"
        #     print predicted_cat[i]
        #     print category_train[i].lower()
    correct_pred = (true / len(train_vector)) * 100
    return correct_pred

print "accuracy is ", accuracy()
