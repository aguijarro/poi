# poi
Final Project - Machine Learning

###### Questions1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is about identify Enron Employees who may have  had committed fraud based on data (financial and email dataset) using machine learning techniques. Machine Learning is useful because it allows us to predict data base in an other related data. In this case, using a techniques to resolve supervised problems, we will identify if a person could be guilty bases in other guilty people who we already know. What it is really important to mention is that data must be process before start the analysis. To do this, it is important to do an outlier investigation to remove certain data like key "Total".

###### Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

For this analysis, the features selected were: 

['to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'shared_receipt_with_poi', 'loan_advances', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'from_poi_to_this_person']. The technique used were SelectKBest with k parameter equal to 10. I chose this parameter because is the half of the len of the dataset. Moreover, the feature scores are: [  1.60054246e-04   1.75169428e+00   2.38995890e-01   3.49627153e-01
   2.28267337e-01   7.79488558e-02   3.13332163e-02   8.90382156e+00
   4.17319228e-03   1.66119123e-01   1.39784138e-02   2.51826104e+00
   1.58770239e-01   6.81945192e-02   2.47052122e+00   5.49084201e-01
   2.19505724e-01   2.22292709e-02   5.44668748e+00]

Also, I applied scaling in two ways. One of them at the begining of the process. It could be considered a general scaling to use with different algorithms, however because the dataset is hugely imbalanced (many more non-POI than POI), the second one must do in each pipeline. Scaling it is important to standarized the values between range [0,1].

To create a new feature, I took as a reference a one entry in the blog, so the new feature created is: ['from_poi']. The idea is to get number related with the to_messages and the from_this_person_poi.


###### Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

The final algorithm I chose is SVM with GridSearchCV because it give me the best precision and recall compare with other algorithms like: Decision Tree and Naive Bayes. Even thought most of the algorithms give me a value in recall equal to 86, just SVM with GridSearchCV give me a better precision value(90).

###### Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tunning the parameters of an algorithm means to found the best values for parameters to get the best perfomance of the algorithm. If you do not do this you probably fall in a mistake like overfitting. In my case, I tune my parameters getting from the svm algorithms the best_estimator_ attribute, and then applying those in the new model.

###### Question 5: What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is the process to test the algorithm in another dataset which is disting of the training dataset. It is important to do this process to avoid a common methodological mistake which is learning the parameters of a prediction function and testing it on the same data. In this project I uses cross-validation strategy to create training and testing datasets.


###### Question 6: Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

In my case, the values obtained for the algorithms are: precision (.90) and recall (.88) in avarage. This parameters means: "precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned" (sklearn site). Here, "Precision and Recall are a useful measure of success of prediction when the classes are very imbalanced.(sklearn site).
