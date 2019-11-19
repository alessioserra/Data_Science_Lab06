'''
Created on 19 nov 2019

@author: zierp
'''
from sklearn.datasets import fetch_openml
from sklearn.model_selection._split import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.classification import classification_report
import math
from sklearn.ensemble import RandomForestClassifier

# EXERCISE 1
dataset = fetch_openml("mnist_784")
X = dataset["data"]
y = dataset["target"]
print("Dataset loaded")
print(len(dataset["data"]))

# EXERCISE 2
tree = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
 
print(classification_report(y_pred, y_test))

# EXERCISE 3
class MyRandomForestClassifier():
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
    
    # Train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        trees = []
        for index in range(self.n_estimators):
            
            tree = DecisionTreeClassifier()
            trees.append(tree)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(self.max_features))
            trees[index].fit(X_train, y_train)
            return trees,y_test,X_test
        
    # Predict the label for each point in X
    def predict(self, X, trees, y_test): 
        
        reports = []
        
        for tree in trees:
            y_predict = tree.predict(X)
            reports.append(classification_report(y_predict, y_test))
            
        return reports    

"""Go method"""
randomForest = MyRandomForestClassifier(10,math.sqrt(len(X)))
trees,y_test,X_test = randomForest.fit(X, y)
reports = randomForest.predict(X_test, trees, y_test)

for report in reports:
    print(report)
    
#EXERCISE 4
"""???"""
rf = RandomForestClassifier(n_estimators = 10, int(max_features=math.sqrt(len(X)))
                            