# poi
Final Project - Machine Learning


###### Questions1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is about identify Enron Employees who may have  had committed fraud based on data (financial and email dataset) using machine learning techniques. Machine Learning is useful because it allows us to predict data base in an other related data. In this case, using a techniques to resolve supervised problems, we will identify if a person could be guilty bases in other guilty people who we already know. What it is really important to mention is that data must be process before start the analysis. 

The dataset is composed by: 146 data points including 1 outlier ("total" key) which will remove. As a result, the final number of data points is 145. This data points are allocation in two classes: POI (18) and No POI (127) this is the reason why the data is unbalance, and accuracy is not a good evaluation metric compared to, say, precision and recall.

The dataset also contains 20 features, most of them with NaN values. The next object show the NaN features and the number of NaN each feature has.

NaN Features:  {'salary': 51, 'to_messages': 60, 'deferral_payments': 107, 'total_payments': 21, 'exercised_stock_options': 44, 'bonus': 64, 'restricted_stock': 36, 'restricted_stock_deferred': 128, 'total_stock_value': 20, 'director_fees': 129, 'from_poi_to_this_person': 60, 'loan_advances': 142, 'from_messages': 60, 'other': 53, 'expenses': 51, 'from_this_person_to_poi': 60, 'deferred_income': 97, 'shared_receipt_with_poi': 60, 'email_address': 35, 'long_term_incentive': 80}




###### Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

New Feature
-----------

Even though the dataset should contain the features needed to make the analysis, some times it is important to identify features, based in human intuition, which might contain some pattern that could exploit using machine learning. In this cases, I will considering the same new feature created in the Lesson 12 (Featured Selection) - Topic: A new Enron Feature. 

In this cases, our intuition shows that it is more common that POIS send emails to other POIS than other people. For that reason, we will get an index with the division between ['to_messages'] variable and ['from_this_person_to_poi']. This new feature will represent the count of messages that are in the person's inbox but are sent to them from person of interest.

To validate how this feature improve or not the dataset we will use kbest algorithm described below.

Select Features
---------------

First of all, to select the features it is very important to present the SelectKBest scores in a readable way. One form to do that is ranking the feature names and their corresponding scores in a descending order. The following list shows the feature names and the scores:

**** Features Scores ****
Feature 1: ('shared_receipt_with_poi', 8.9038215571655712)
Feature 2: ('from_poi_to_this_person', 5.4466874833253529)
Feature 3: ('loan_advances', 2.5182610445203437)
Feature 4: ('from_this_person_to_poi', 2.470521222656084)
Feature 5: ('to_messages', 1.7516942790340737)
Feature 6: ('director_fees', 0.54908420147980874)
Feature 7: ('total_payments', 0.34962715304280179)
Feature 8: ('deferral_payments', 0.23899588985313305)
Feature 9: ('exercised_stock_options', 0.22826733729104948)
Feature 10: ('deferred_income', 0.21950572394230994)
Feature 11: ('total_stock_value', 0.16611912320976677)
Feature 12: ('from_messages', 0.1587702392129193)
Feature 13: ('bonus', 0.077948855777229875)
Feature 14: ('other', 0.068194519159558625)
Feature 15: ('restricted_stock', 0.031333216297618476)
Feature 16: ('from_poi', 0.028110269917615248)
Feature 17: ('long_term_incentive', 0.022229270861607336)
Feature 18: ('expenses', 0.01397841382175243)
Feature 19: ('restricted_stock_deferred', 0.0041731922805086684)
Feature 20: ('salary', 0.00016005424569618399)

As we can see, after the Feature 5 the scores drop off, so for that reason the top 5 features are the most important ones. As a result the features selected are:

Features Selected:  ['shared_receipt_with_poi', 'from_poi_to_this_person', 'loan_advances', 'from_this_person_to_poi', 'to_messages']

It is important to mention that the new feature will not considered because the score is too slow: Feature 16: ('from_poi', 0.028110269917615248)

Also, I applied scaling in two ways. One of them at the begining of the process. It could be considered a general scaling to use with different algorithms, however because the dataset is hugely imbalanced (many more non-POI than POI), the second one must do in each pipeline. Scaling it is important to standarized the values between range [0,1].



###### Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

The final algorithm I chose is SVM with GridSearchCV because it give me the best precision and recall compare with other algorithms like: Decision Tree and Naive Bayes. Even thought most of the algorithms give me a value in recall equal to 86, just SVM with GridSearchCV give me a better precision value(90).

###### Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tunning the parameters of an algorithm means to found the best values for parameters to get the best perfomance of the algorithm. If you do not do this you probably have a model with low performance for both training and testing data and the data could not be classify appropriately.
As a result, the cause of poor performance in machine learning is either overfitting or underfitting the data. 

"Overfitting refers to a model that models the training data too well. It learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize." Underfitting refers to a model that can neither model the training data nor generalize to new data. ("Overfitting and Underfitting With Machine Learning Algorithms",  Jason Brownlee on March 21, 2016 in Machine Learning Algorithms)

In my case, I tune my parameters getting from the svm algorithms the best_estimator_ attribute, and then applying those in the new model.

###### Question 5: What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is the process to test the algorithm in another dataset which is disting of the training dataset. It is important to do this process to avoid a common methodological mistake which is learning the parameters of a prediction function and testing it on the same data. In this project I uses cross-validation strategy to create training and testing datasets. Scikit-learn has a collection of classes which can be used to generate lists of train/test indices for popular cross-validation strategies. In this project, the method uses is train_test_split.

###### Question 6: Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

In my case, the values obtained for the algorithms are: precision (.90) and recall (.88) in avarage. This parameters means: "precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned" (sklearn site). Here, "Precision and Recall are a useful measure of success of prediction when the classes are very imbalanced.(sklearn site).

In our case, because precision says how exact you were among identify each class, the precision to identify NoPois was around 92% which means that 24(NoPois) of 26( 24 NoPois + 2 Pois) classes were correctly identify. In the cases of POIS the precision was 0. Meanwhile, Recall says, how complete you were among your identify each class. So, for NoPois was around 1, what means that every NoPois of the total of NoPois where identify. For Pois, was 0.