# importing the required packages
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# importing the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# function to detect outliers in the data
def detect_outliers(df,n,features):
   
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers 

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

# Dropping detected outliers from the training dataset
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# Merging both the datasets to apply feature engineering
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
dataset["Age"] = dataset["Age"].fillna(dataset["Fare"].mean()).astype(int)
dataset = pd.get_dummies(dataset,columns = ['Sex', 'Embarked'], drop_first = True)
dataset.drop('PassengerId', axis = 1, inplace = True)

dataset['Name'] = dataset['Name'].apply(lambda x : x.split(',')[1].split('.')[0])
row = sorted([29,147,391,549,148,244,619,839,876,1045,1030,240,312,319,625,653,759,788,1174,442,529,592,640,687,1012,1083,738,752,814,1295])
dataset.iloc[row,3] = ' Other'

d = {' Master':0, ' Miss':1, ' Ms' : 1 , ' Mme':1, ' Mlle':1, ' Mrs':1, ' Mr':2, ' Other':3} #
l = list(dataset['Name'])
Title = [d[x] for x in l]
dataset['Title'] = Title

dataset['fam_size'] = 0
dataset['fam_size'] = pd.Series(dataset['SibSp'] + dataset['Parch'])

dataset['Age_band']=0
dataset.loc[dataset['Age']<=16,'Age_band']=0
dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age_band']=1
dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age_band']=2
dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age_band']=3
dataset.loc[dataset['Age']>64,'Age_band']=4

dataset['Alone']=0
dataset.loc[dataset['fam_size']==0,'Alone']=1

dataset['Fare_categ']=0
dataset.loc[dataset['Fare']<=7.91,'Fare_categ']=0
dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare_categ']=1
dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare_categ']=2
dataset.loc[(dataset['Fare']>31)&(dataset['Fare']<=513),'Fare_categ']=3

dataset.drop(['Name','Fare','Age','fam_size','Cabin','Ticket','SibSp','Parch'], axis = 1, inplace = True)

# Splitting back the merged dataset into training set and test set
train = dataset.iloc[range(881),:]
indep = train.iloc[:,[0,2,3,4,5,6,7,8]] #independent features in training set
dep = train['Survived'] #dependent feature in training set
test_set = dataset.iloc[range(881,1299),:]
test_set.drop(['Survived'],axis = 1, inplace = True)

# Splitting the training and test sets for further evaluation
(indep_train, indep_test, dep_train, dep_test) = train_test_split(indep,dep,test_size = 0.2, random_state = 0)

# Checking out the best classifiers on the transformed data
classifiers = []
classifiers.append(SVC(random_state = 0))
classifiers.append(DecisionTreeClassifier(random_state = 0))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = 0),random_state = 0))
classifiers.append(RandomForestClassifier(random_state = 0))
classifiers.append(ExtraTreesClassifier(random_state = 0))
classifiers.append(GradientBoostingClassifier(random_state = 0))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = 0))
classifiers.append(XGBClassifier())

accuracies = []
for classifier in classifiers :
    accuracies.append(cross_val_score(classifier, indep_train, dep_train, scoring = 'accuracy', cv = 10, n_jobs= -1))

mean_acc = []
std_acc= []
for i in accuracies:
    mean_acc.append(i.mean())
    std_acc.append(i.std())

# Performing Gird Search for parameter tuning of best performing classifiers 
from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [3,4,5,6], 'learning_rate' : [0.1,0.2,0.3,0.4,0.5,0.6], 'n_estimators' : [100,120,140,160,180], 'gamma' : [0,0.1,0.2,0.3,0.4,0.5,0.6]}
gs = GridSearchCV(estimator = XGBClassifier(), param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = 10)
gs = gs.fit(indep_train, dep_train)
best_accuracy = gs.best_score_
best_params = gs.best_params_

from sklearn.model_selection import GridSearchCV
params = {'C' : [0.5,1.0,1.5,2.0,5.0,10.0], 'kernel' : ['linear', 'rbf'],'gamma' : [0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]}
gs1 = GridSearchCV(estimator = SVC(probability = True), param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = 10)
gs1 = gs1.fit(indep_train, dep_train)
best_accuracy = gs1.best_score_
best_params = gs1.best_params_

from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [2,3,4,5,6],'min_samples_split' :[2,3,4,5], 'min_samples_leaf':[1,2,3,4],'max_leaf_nodes' : [2,3,4,5],'bootstrap':[True,False],'n_estimators' : [10,20,40,50,100,120]}
gs2 = GridSearchCV(estimator = RandomForestClassifier(random_state = 0, n_jobs = -1), param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = 10)
gs2 = gs2.fit(indep_train, dep_train)
best_accuracy2 = gs2.best_score_
best_params2 = gs2.best_params_

from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [2,3,4,5,6], 'learning_rate' : [0.2,0.3,0.4,0.5], 'n_estimators' : [50,100,120,140,160,180],'min_samples_split': [2,3,4],'min_samples_leaf':[1,2,3],'max_features':[2,3,4,5]}
gs3 = GridSearchCV(estimator = GradientBoostingClassifier(random_state = 0), param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = 10)
gs3 = gs3.fit(indep_train, dep_train)
best_accuracy3 = gs3.best_score_
best_params3 = gs3.best_params_

# Creating an Ensemble Classifier
ensemble = VotingClassifier(estimators=[('svc',SVC(random_state = 0,probability=True,C=0.1,gamma = 0.2,kernel='rbf')),('rfc', RandomForestClassifier(n_estimators = 120 ,max_depth = 2,max_leaf_nodes = 3,random_state = 0, n_jobs = -1,)),('xgb',XGBClassifier(n_estimators = 120,gamma = 0.2)),('gbc',GradientBoostingClassifier(learning_rate = 0.3,n_estimators=180,min_samples_leaf=2,max_features=3,random_state = 0))],voting='soft')

# Fitting the Ensemble Classifier
ensemble.fit(indep_train,dep_train)

# Predicting the results
y_pred = ensemble.predict(indep_test).astype(int)
cm = confusion_matrix(dep_test, y_pred)

predictions = pd.DataFrame(ensemble.predict(test_set)).astype(int)
Id = pd.Series(test.loc[:,'PassengerId'])
predictions = pd.concat([Id,predictions], axis = 1)
predictions = predictions.rename(columns = {0 :'Survived'})
predictions.to_csv("Results.csv",index=False)
