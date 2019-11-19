'''
Created on 19 nov 2019

@author: zierp
'''
import csv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pandas as pd

# EXERCISE 1
data = []
with open("2dWine.csv") as f:
    for row in csv.reader(f):
        line = []
        for el in row:
            line.append(el)
         
        #list of lists   
        data.append(line)

data.pop(0)

x0 = []
y0 = []
x1 = []
y1 = []

for row in data:
    
    if row[2]=='0':     
        x0.append(row[0])
        y0.append(row[1])
    
    if row[2]=='1':     
        x1.append(row[0])
        y1.append(row[1])

"""Scanner plot"""
plt.scatter(x0, y0, c='red', alpha=0.5)
plt.scatter(x1, y1, c='blue', alpha=0.5)
plt.show()

# EXERCISE 2
#READING FILE WITH PANDAS
columns_name = ['x0', 'x1', 'label']
d = pd.read_csv("2dWine.csv", header=0, names=columns_name, ) # NOTE: header=0 skip headers
col = ['x0', 'x1']

X = d[col] #FEATURES
y = d.label #LABEL

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
dot_code = export_graphviz(clf, feature_names=col)
"""Copy and paste the string of dot_code a the following link to visualize the graph:
http://www.webgraphviz.com/
"""
print(dot_code)
        
