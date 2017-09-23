import math
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
# %matplotlib inline

def process_raw_df(df):        
    df_data = df.drop(['PassengerId', 'Age', 'Ticket', 'Cabin', 'Name'], axis=1)
    
    df_data['HasCabin'] = df['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
    df_data['Embarked'] = df_data['Embarked'].fillna('S')
    df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())
    
    try:
        se_Y = df_data['Survived']
        df_data = df_data.drop(['Survived'], axis=1)
    except KeyError:
        se_Y = None        
    
    df_data = pd.get_dummies(df_data).drop(['Sex_female', 'Embarked_S'], axis=1)
    return df_data, se_Y

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_Xtrain, se_Ytrain = process_raw_df(df_train)
df_Xtest, _ = process_raw_df(df_test)

##############################################################################################
# Adaboost
##############################################################################################
'''
n_leafs = list(range(3,9))
n_trees = []
adb_clfs = [] 
for n_leaf in n_leafs:
    # for each n_leaf, find the best number of trees n_tree
    adb = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_leaf_nodes = n_leaf))
    n_tree_grid = {'n_estimators': list(range(50,1050,50))}
    gcv = GridSearchCV(adb, param_grid=n_tree_grid, n_jobs=-1, cv=5)
    gcv.fit(df_Xtrain, se_Ytrain)
    # best_params_ is a dict like so {'n_estimators': 75} 
    n_trees.append(gcv.best_params_['n_estimators'])
    adb_clfs.append(gcv.best_estimator_)
    
# now score the adb_clfs using Repeated K-Fold
score_means = []
score_stds = []
for n_leaf, n_tree, adb_clf in zip(n_leafs, n_trees, adb_clfs):
    scores = cross_val_score(adb_clf, df_Xtrain, se_Ytrain, cv=5, n_jobs=-1)
    score_means.append(scores.mean())
    score_stds.append(scores.std())

# plot Adaboost result
plt.figure()
x_tickmarks = ['nleafs: {0}, ntrees: {1}'.format(i, j) for i, j in zip(n_leafs, n_trees)]
plt.scatter(range(len(x_tickmarks)), score_means, marker='o', label='mean of cv scores')
plt.scatter(range(len(x_tickmarks)), score_stds, marker='X', label='std of cv scores')
plt.legend(loc='best')
plt.title('Scoring Adaboost models')
plt.xlabel('Model parameters')
plt.xticks(range(len(x_tickmarks)), x_tickmarks, rotation='vertical')
plt.margins(0.2)
plt.show()
'''
##############################################################################################
# SVM
##############################################################################################
svm_param_grid = {'kernel': ['rbf', 'poly', 'sigmoid'],
                  'C': np.logspace(-2, 0, num=8, base=round(math.exp(1), 2)),
                  'gamma': np.logspace(-4, 4, num=8, base=round(math.exp(1), 2))}
gcv = GridSearchCV(svm.SVC(), param_grid=svm_param_grid, cv=5)
gcv.fit(df_Xtrain, se_Ytrain)
best_svm = gcv.best_estimator_

predictions = best_svm.predict(df_Xtest)
df_submit = pd.DataFrame({'PassengerId': df_test['PassengerId'], 
                          'Survived': predictions})
df_submit.to_csv('submit_svm.csv', index=False)
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  