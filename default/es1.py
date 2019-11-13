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

# Exercise 2.1
dataset = load_wine()
X = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]
print("Computed")

# Exercise 2.2
clf = DecisionTreeClassifier()

# Exercise 2.3
clf = clf.fit(X, y)
dot_code = export_graphviz(clf, feature_names=feature_names)
"""Copy and paste the string of dot_code a the following link to visualize the graph:
http://www.webgraphviz.com/
"""
print(dot_code)

# Exercise 2.4
y_test_pred = clf.predict(X)
acc = accuracy_score(y, y_test_pred)
print("\nAccuracy score is: "+str(acc)+"\n")

# Exercise 2.5-6
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

# Exercise 2.7
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
    
# Exercise 2.8

