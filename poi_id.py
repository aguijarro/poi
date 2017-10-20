#!/usr/bin/python

import sys
import pickle
from time import time
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


def get_Features(features, labels):
    kbest = SelectKBest(k=10)
    selected_features = kbest.fit_transform(features, labels)
    features_selected=[features_list_selection[i+1] for i in kbest.get_support(indices=True)]
    print "Features Selected Scores:"
    print kbest.scores_
    print "Features Selected Names:"
    print features_selected
    return features_selected


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

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_selection = ['poi','salary','to_messages','deferral_payments', 'total_payments',
                           'exercised_stock_options', 'bonus', 'restricted_stock',
                           'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value',
                           'expenses', 'loan_advances', 'from_messages', 'other',
                           'from_this_person_to_poi', 'director_fees', 'deferred_income',
                           'long_term_incentive', 'from_poi_to_this_person']# You will need to use more features


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




### Store to my_dataset for easy export below.
my_dataset_selection = data_dict

### Extract features and labels from dataset for local testing
data_selection = featureFormat(my_dataset_selection, features_list_selection, sort_keys = True)
labels_selection, features_selection = targetFeatureSplit(data_selection)

### Task 1: Select what features you'll use.

features_selected = get_Features(features_selection, labels_selection)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] + features_selected

### Task 2: Remove outliers

key = 'TOTAL'
data_dict.pop(key, 0)

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

#### Naive Bayes Classifier
# import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB
# create classifier
clf_nb = GaussianNB()

#### Naive Bayes Classifier with pipeline
# create a pipeline with scaling and dimension reduction
pipeline_nb = Pipeline([('scaler', MinMaxScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf_nbp', GaussianNB()),
                        ])

#### DecisionTree Classifier
from sklearn.grid_search import GridSearchCV
# import the sklearn module for tree
from sklearn.tree import DecisionTreeClassifier
# create classifier
clf_dt = DecisionTreeClassifier(min_samples_split=40)


#### DecisionTree with Pipeline
# create a pipeline with scaling and dimension reduction
pipeline_tree = Pipeline([('selection', SelectKBest(k=10)),
                          ('scaler', MinMaxScaler()),
                          ('pca', PCA(n_components=2)),
                          ('clf_tree', DecisionTreeClassifier(min_samples_split=40)),
                          ])

#### DecisionTree with Pipeline and GridSearchCV
# create a pipeline with scaling and dimension reduction
pipeline_tree_1 = Pipeline([('scaler_tr1', MinMaxScaler()),
                          ('pca_tr1', PCA()),
                          ('clf_tree_tr1', DecisionTreeClassifier()),
                          ])

parameters_tree_1 = {
    'pca_tr1__n_components': [2],
    'clf_tree_tr1__min_samples_split': [40],
}
clf_tree_gs = GridSearchCV(pipeline_tree_1, parameters_tree_1, verbose = 1, scoring = "roc_auc")


#### SVM with dimension reduction
# import the sklearn module for SVC

from sklearn.svm import SVC
param_grid = {
              'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_svmpca = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)



param_grid_def = {
                  'C': [50000.0],
                  'gamma': [0.01],
                  'cache_size' : [200],
                  'class_weight' : ['balanced'],
                  'coef0' : [0.0],
                  'decision_function_shape' : [None],
                  'degree':[3],
                  'kernel':['rbf'],
                  'max_iter':[-1],
                  'probability':[False],
                  'random_state':[None],
                  'shrinking':[True],
                  'tol':[0.001],
                  'verbose':[False]
                  }
clf_svmpca_def = GridSearchCV(SVC(), param_grid)


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

#### Naive Bayes Classifier Fitting
print "***** Naive Bayes Classifier Fitting *****"

t0_nb = 0
clf_nb.fit(features_train, labels_train)
print "training time:", round(time() - t0_nb, 3), "s"
t1_nb = time()
pred_nb = clf_nb.predict(features_test)
print "predicting time clf_nb:", round(time() - t1_nb, 3), "s"
acc_nb= getAccuracy(pred_nb, labels_test)
print "Accuracy clf_nb", acc_nb


#### Naive Bayes Classifier with pipeline fitting
print "***** Naive Bayes Classifier with pipeline fitting *****"
pipeline_nb.fit(features_train,labels_train)
pipeline_nb_pred = pipeline_nb.predict(features_test)
pipelin_nb_acc = getAccuracy(pipeline_nb_pred, labels_test)
print "Accuracy clf_nb", pipelin_nb_acc
print classification_report(labels_test, pipeline_nb_pred)
print confusion_matrix(labels_test, pipeline_nb_pred)


#### DecisionTree
print "***** DecisionTree fitting *****"
t0_dt = time()
clf_dt.fit(features_train, labels_train)
print "training time:", round(time() - t0_dt, 3), "s"
t1_dt = time()
pred_dt = clf_dt.predict(features_test)
print "predicting time:", round(time() - t1_dt, 3), "s"
acc_dt = getAccuracy(pred_dt, labels_test)
print "Accuracy_dt", acc_dt

#### DecisionTree Classifier with pipeline fitting

print "***** DecisionTree Classifier with pipeline fitting *****"

pipeline_tree.fit(features_train,labels_train)
pipeline_tree_pred = pipeline_tree.predict(features_test)
pipelin_tree_acc = getAccuracy(pipeline_tree_pred, labels_test)
print "Accuracy clf_tree", pipelin_tree_acc
print classification_report(labels_test, pipeline_tree_pred)
print confusion_matrix(labels_test, pipeline_tree_pred)


#### DecisionTree Classifier with GridSearchCV
print "***** DecisionTree Classifier with GridSearchCV *****"
print("\nPerforming grid search...")
print("pipeline:", [name for name, _ in pipeline_tree_1.steps])
print("parameters:")
print(parameters_tree_1)
print "Fitting the classifier to the training set"
t0_tree_gs = time()
clf_tree_gs.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0_tree_gs)
print "Best estimator found by grid search:"
print clf_tree_gs.best_estimator_

t1_tree_gs = time()
pred_tree_gs = clf_tree_gs.predict(features_test)
print "done in %0.3fs" % (time() - t1_tree_gs)
print classification_report(labels_test, pred_tree_gs)
print confusion_matrix(labels_test, pred_tree_gs)
acc_tree_gs = getAccuracy(pred_tree_gs, labels_test)
print "Accuracy", acc_tree_gs


#### SVM with GridSearchCV
# Train a SVM classification model
print "***** SVM with GridSearchCV *****"
print "Fitting the classifier to the training set"
t0_svmpca = time()
clf_svmpca.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0_svmpca)
print "Best estimator found by grid search:"
print clf_svmpca.best_estimator_
# Quantitative evaluation of the model quality on the test set
print "Predicting the people names on the testing set"
t1_svmpca = time()
pred_svmpca = clf_svmpca.predict(features_test)
print "done in %0.3fs" % (time() - t1_svmpca)
print classification_report(labels_test, pred_svmpca)
print confusion_matrix(labels_test, pred_svmpca)
acc_svmpca = getAccuracy(pred_svmpca, labels_test)
print "Accuracy", acc_svmpca


#### SVM with GridSearchCV Tunning
# Train a SVM classification model
print "***** SVM with GridSearchCV Tunning*****"
print "Fitting the classifier to the training set"
t0_svmpca_def = time()
clf_svmpca_def.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0_svmpca_def)

# Quantitative evaluation of the model quality on the test set
print "Predicting the people names on the testing set"
t1_svmpca_def = time()
pred_svmpca_def = clf_svmpca_def.predict(features_test)
print "done in %0.3fs" % (time() - t1_svmpca_def)
print classification_report(labels_test, pred_svmpca_def)
print confusion_matrix(labels_test, pred_svmpca_def)
acc_svmpca_def = getAccuracy(pred_svmpca_def, labels_test)
print "Accuracy", acc_svmpca_def


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_svmpca_def, my_dataset, features_list)