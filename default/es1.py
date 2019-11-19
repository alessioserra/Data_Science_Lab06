'''
Created on 13 nov 2019

@author: zierp
'''
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection._split import train_test_split
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

# Exercise 1
dataset = load_wine()
X = dataset["data"]
print(X[0])
y = dataset["target"]
feature_names = dataset["feature_names"]
print("Computed")

# Exercise 2
clf = DecisionTreeClassifier()

# Exercise 3
clf = clf.fit(X, y)
dot_code = export_graphviz(clf, feature_names=feature_names)
"""Copy and paste the string of dot_code a the following link to visualize the graph:
http://www.webgraphviz.com/
"""
print(dot_code)

# Exercise 4
y_test_pred = clf.predict(X)
acc = accuracy_score(y, y_test_pred)
print("\nAccuracy score is: "+str(acc)+"\n")

# Exercise 5-6
"""Split datas"""
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
"""New tree"""
tree2 = DecisionTreeClassifier()
"""fit model"""
tree2.fit(X_train, y_train)
"""predict the values of the model passing the X's"""
y_pred = tree2.predict(X_test) 
"""Compare y_'fitted' vs. y_predicted with classification_report"""
print(classification_report(y_pred, y_test))

# Exercise 7
params = {
"max_depth": [None, 2, 4, 8],
"splitter": ["best", "random"]
}

for config in ParameterGrid(params):
    clf = DecisionTreeClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    print("Classification report: with",config,"\n")
    print(classification_report(y_pred, y_test))
    
# Exercise 8
"""k-fold cross-validation"""

"""Split the datasets into two:
# - X_train_valid: the dataset used for the k-fold cross-validation
# - X_test: the dataset used for the final testing (this will NOT
# be seen by the classifier during the training/validation phases)"""
# Train test splitting
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, train_size=0.5)

kf = KFold(5) # 5-fold cross-validation
""""X and y are the arrays to be split"""
for train_indices, validation_indices in kf.split(X_train_valid):
    X_train = X_train_valid[train_indices]
    X_valid = X_train_valid[validation_indices]
    y_train = y_train_valid[train_indices]
    y_valid = y_train_valid[validation_indices]
    
    for config in ParameterGrid(params):
        clf = DecisionTreeClassifier(**config)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid) 
        print("Classification report K-FOLD: with",config,"\n")
        print(classification_report(y_pred, y_valid))
