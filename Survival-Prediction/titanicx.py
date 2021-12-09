import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
indep = dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
dep = dataset.iloc[:,1]
test = testset.iloc[:,1:]

title = []
for row in test.loc[:,'Name']:
    title.append(row.split(',')[1].split('.')[0])
Title = pd.Series(title)
fam_size = test.loc[:,'SibSp'].add(test.loc[:,'Parch'])
dona = pd.Series(np.zeros(shape=(len(indep))))
test = pd.concat([test,fam_size,Title], axis = 1)
test = test.rename(columns = {0 :'Fam_Size', 1 : 'Title'})
test = test.drop(labels = ['SibSp','Parch','Ticket','Name'], axis = 1)
test['Age'] = test.loc[:,'Age'].fillna(value = test['Age'].mean()).astype('int64')
test = pd.get_dummies(test,columns = ['Sex', 'Embarked','Title'], drop_first = True)
test['Cabin'] = test.loc[:,'Cabin'].notnull().astype('int64')


title = []
for row in indep.loc[:,'Name']:
    title.append(row.split(',')[1].split('.')[0])
Title = pd.Series(title)
dona = pd.Series(np.zeros(shape=(len(indep))))
fam_size = indep.loc[:,'SibSp'].add(indep.loc[:,'Parch'])
indep = pd.concat([indep,fam_size,Title,dona], axis = 1)
indep = indep.rename(columns = {0 :'Fam_Size', 1 :'Title', 2 : 'Title_ Dona'})
indep = indep.drop(labels = ['SibSp','Parch','Ticket','Name'], axis = 1)
indep['Age'] = indep.loc[:,'Age'].fillna(value = indep['Age'].mean()).astype('int64')
indep['Embarked'] = indep.loc[:,'Embarked'].fillna(value = indep['Embarked'].mode()[0])
indep = pd.get_dummies(indep,columns = ['Sex', 'Embarked','Title'], drop_first = True)
indep['Cabin'] = indep.loc[:,'Cabin'].notnull().astype('int64')

indep = indep.loc[:,list(test)]

from sklearn.model_selection import train_test_split
(indep_train, indep_test, dep_train, dep_test) = train_test_split(indep,dep,test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier
model = XGBClassifier()

from sklearn.model_selection import cross_val_score
kf = cross_val_score(model, indep_train, dep_train, cv = 10, n_jobs = -1)
kf.mean()
kf.std()
model.fit(indep_train, dep_train)
y_pred = model.predict(indep_test)
#model.fit(indep_train, dep_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dep_test, y_pred)

predictions = pd.DataFrame(model.predict(test))
Survived = pd.Series(testset.loc[:,'PassengerId'])
predictions = pd.concat([Survived,predictions], axis = 1)

predictions = pd.DataFrame.to_csv('Submission.csv')
#predictions = pd.DataFrame.to_csv('Survival.csv')

'''from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [3,4,5,6], 'learning_rate' : [0.1,0.2,0.3,0.4,0.5,0.6], 'n_estimators' : [100,120,140,160,180,200], 'gamma' : [0,0.1,0.2,0.3,0.4,0.5,0.6]}
gs = GridSearchCV(estimator = model, param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = 10)
gs = gs.fit(indep_train, dep_train)
best_accuracy = gs.best_score_
best_params = gs.best_params_'''






