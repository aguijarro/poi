#!/usr/bin/python

import sys
import pickle
from time import time
from collections import OrderedDict
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def get_Features(features, labels):
    kbest = SelectKBest(k = 'all')
    selected_features = kbest.fit_transform(features, labels)

    features_selected=[features_list_selection[i+1] for i in kbest.get_support(indices=True)]

    score_names = {}
    j = 0
    for i in features_selected:
        score_names[i] = kbest.scores_[j]
        j += 1

    name_scores = sorted(score_names.items(), key=lambda x: x[1], reverse=True)

    j = 1
    print "**** Features Scores ****"
    for n in name_scores:
        print "Feature %d:" % j, n
        j += 1

    list_features = []
    for n in name_scores:
        if n[1] >= 1:
            list_features.append(n[0])

    print "Features Selected: ", list_features
    return list_features


def featureScaling(features):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    rescaled_feature = scaler.fit_transform(features)
    return rescaled_feature

def dimension_reduction(features):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(features)
    #return pca.components_[0], pca.components_[1]
    return pca

def getAccuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 3: Create new feature(s)

## Code taken from a post in Udacity forum. (https://goo.gl/JkJyv4)

for employee in data_dict:
    if (data_dict[employee]['to_messages'] not in ['NaN', 0]) and (data_dict[employee]['from_this_person_to_poi'] not in ['NaN', 0]):
        data_dict[employee]['from_poi'] = float(data_dict[employee]['to_messages'])/float(data_dict[employee]['from_this_person_to_poi'])
    else:
        data_dict[employee]['from_poi'] = 0


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_selection = ['poi','salary','to_messages','deferral_payments', 'total_payments',
                           'exercised_stock_options', 'bonus', 'restricted_stock',
                           'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value',
                           'expenses', 'loan_advances', 'from_messages', 'other',
                           'from_this_person_to_poi', 'director_fees', 'deferred_income',
                           'long_term_incentive', 'from_poi_to_this_person', 'from_poi']# You will need to use more features


### Store to my_dataset for easy export below.
my_dataset_selection = data_dict

### Extract features and labels from dataset for local testing
data_selection = featureFormat(my_dataset_selection, features_list_selection, sort_keys = True)
labels_selection, features_selection = targetFeatureSplit(data_selection)


### Dataset Description
print "**** Dataset Description ****"
print "Total number of data points: ", len(data_dict)
print "Number of features: 20 features"
print "*** labels_selection: ", len(labels_selection)

## Allocation classes

class_0 = 0
class_1 = 0
for n in labels_selection:
    if n == 0:
        class_0 = class_0 + 1
    else:
        class_1 = class_1 + 1

print "allocation across classes"
print "Number of POI: ", class_1
print "Number of no POI: ", class_0

## Identify NaN values.

nan_features = {}

print "Identify NaN values"
for key, value in data_dict.iteritems():
    for key_val, val_val in value.iteritems():
        if val_val == 'NaN':
            if key_val in nan_features.keys():
                nan_features[key_val] = nan_features[key_val] + 1
            else:
                nan_features[key_val] = 1

print "NaN Features: ", nan_features


### Task 1: Select what features you'll use.


features_selected = get_Features(features_selection, labels_selection)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] + features_selected

### Task 2: Remove outliers


data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scaling features
features = featureScaling(features)

### Dimension Reduction
pca = dimension_reduction(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#### Naive Bayes Classifier - clf_nb
print "**** Creating Naive Bayes Classifier - clf_nb ****"
clf_nb = GaussianNB()

#### Naive Bayes Classifier Pipeline - clf_nbp
print "**** Creating Naive Bayes Classifier Pipeline - clf_nbp ****"
clf_nbp = GaussianNB()
scaler_nbp = MinMaxScaler()
pca_nbp = PCA()
pipeline_nbp = Pipeline([('scaler_nbp', scaler_nbp ),
                        ('pca_nbp', pca_nbp),
                        ('clf_nbp', clf_nbp),
                         ])
parameters_nbp = {
    'pca_nbp__n_components': [2],
}
gs_nbp = GridSearchCV(pipeline_nbp, param_grid=parameters_nbp, scoring='f1')


#### DecisionTree Classifier - clf_dt
print "**** Creating DecisionTree Classifier - clf_dt ****"
clf_dt = DecisionTreeClassifier(min_samples_split=40)

#### DecisionTree Classifier - clf_tree
print "**** Creating DecisionTree Classifier - clf_tree ****"

clf_tree = DecisionTreeClassifier()
selection_tree = SelectKBest()
scaler_tree = MinMaxScaler()
pca_tree = PCA()

pipeline_tree = Pipeline([('selection_tree', selection_tree),
                          ('scaler_tree', scaler_tree),
                          ('pca_tree', pca_tree),
                          ('clf_tree', clf_tree),
                          ])

parameters_tree = {
    'selection_tree__k': [5],
    'pca_tree__n_components': [2],
    'clf_tree__min_samples_split': [40],
}

clf_tree_gs = GridSearchCV(pipeline_tree, parameters_tree, verbose=1, scoring="f1")

#### SVM Classifier - clf_svm
print "**** Creating SVM Classifier - clf_svm ****"
param_grid = {
              'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
clf_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)


#### SVM Classifier - clf_svc_t
print "**** Creating SVM Classifier - clf_svc_t ****"
clf_svc_t = SVC()
kbest_svc_t = SelectKBest()
scaler_svc_t = MinMaxScaler()
pipe_svc_t = Pipeline([('kbest_svc_t', kbest_svc_t),
                      ('scaler_svc_t', scaler_svc_t),
                      ('clf_svc_t', clf_svc_t)
                      ])
param_grid_def = {'kbest_svc_t__k': [5],
                  'clf_svc_t__C': [1000.0],
                  'clf_svc_t__gamma': [0.001],
                  'clf_svc_t__cache_size' : [200],
                  'clf_svc_t__class_weight' : ['balanced'],
                  'clf_svc_t__coef0' : [0.0],
                  'clf_svc_t__decision_function_shape' : [None],
                  'clf_svc_t__degree':[3],
                  'clf_svc_t__kernel':['rbf'],
                  'clf_svc_t__max_iter':[-1],
                  'clf_svc_t__probability':[False],
                  'clf_svc_t__random_state':[None],
                  'clf_svc_t__shrinking':[True],
                  'clf_svc_t__tol':[0.001],
                  'clf_svc_t__verbose':[False]
                  }


#### Naive Bayes Classifier - Tunning - clf_gnb8
print "**** Creating Naive Bayes Classifier - Tunning - clf_gnb8 ****"
clf_gnb8 = GaussianNB()
kbest_gnb8 = SelectKBest()
scaler_gnb8 = MinMaxScaler()
pipe_gnb8 = Pipeline([('kbest_gnb8', kbest_gnb8),
                      ('scaler_gnb8', scaler_gnb8),
                      ('clf_gnb8', clf_gnb8)
                      ])
parameters = {'kbest_gnb8__k': [5]}


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#### Naive Bayes Classifier Fitting - clf_nb
print "***** Fitting Naive Bayes Classifier Fitting - clf_nb *****"
clf_nb.fit(features_train, labels_train)

#### Naive Bayes Classifier Fitting Pipeline - clf_nbp
print "***** Fitting Naive Bayes Classifier Fitting Pipeline - clf_nbp *****"
gs_nbp.fit(features_train, labels_train)
clf_gs_nbp_be = gs_nbp.best_estimator_

#### DecisionTree Classifier Fitting - clf_dt
print "***** Fitting DecisionTree Classifier Fitting - clf_dt *****"
clf_dt.fit(features_train, labels_train)

#### DecisionTree Classifier Fitting - clf_tree
print "***** Fitting DecisionTree Classifier - clf_tree *****"
clf_tree_gs.fit(features_train, labels_train)
clf_tree_be = clf_tree_gs.best_estimator_


#### SVM Classifier Fitting - clf_svm
print "***** Fitting SVM Classifier Fitting - clf_svm *****"
clf_svm.fit(features_train, labels_train)
clf_svm_be = clf_svm.best_estimator_
print "Best Estimators SVM Classifier"
print clf_svm_be


#### SVM Classifier Fitting - clf_svc_t

print "***** Fitting SVM with GridSearchCV Tunning *****"
sk_fold_svc_t = StratifiedShuffleSplit(labels_train, 100, random_state=42)
gs_svc_t = GridSearchCV(pipe_svc_t, param_grid=param_grid_def, cv=sk_fold_svc_t, scoring='f1')
gs_svc_t.fit(features, labels)
clf_svc_t_be = gs_svc_t.best_estimator_

#### Naive Bayes Classifier Fitting - Tunning - clf_gnb8
print "***** Fitting GaussianNB Tunning *****"
sk_fold_gnb8 = StratifiedShuffleSplit(labels, 1000, random_state=42)
gs_gnb8 = GridSearchCV(pipe_gnb8, param_grid=parameters, cv=sk_fold_gnb8, scoring='f1')
gs_gnb8.fit(features, labels)
clf_gnb8_be = gs_gnb8.best_estimator_

print "Best Estimator Fitting SVM with GridSearchCV Tunning"
print clf_gnb8_be


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf_nb, my_dataset, features_list_selection)
#dump_classifier_and_data(clf_gs_nbp_be, my_dataset, features_list_selection)
#dump_classifier_and_data(clf_dt, my_dataset, features_list_selection)
#dump_classifier_and_data(clf_tree_be, my_dataset, features_list_selection)
#dump_classifier_and_data(clf_svm_be, my_dataset, features_list_selection)
#dump_classifier_and_data(clf_svc_t_be, my_dataset, features_list_selection)
dump_classifier_and_data(clf_gnb8_be, my_dataset, features_list_selection)

