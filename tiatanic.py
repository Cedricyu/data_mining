
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test["PassengerId"]

print(type(train))
train.head()

train.info

train.describe()
test.info()

train.duplicated().sum()

train.isna().sum().sort_values(ascending=False)

100* (train.isnull().sum() / len(train)).sort_values(ascending=False)
train.corr()['Survived']

import matplotlib.pyplot as plt
import seaborn as sns

train_num = train[['Age','SibSp','Parch','Fare']]
train_cat = train[['Survived','Pclass','Sex',"Ticket",'Cabin','Embarked']]

plt.figure(figsize=(5,3), dpi=150)
sns.heatmap(train_num.corr())
def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)
    
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("U", inplace=True)
    return data


train = clean(train)
test = clean(test)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


cols = ["Sex","Embarked"]

for col in cols:
    train [col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    
train.head(5)


def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)
    
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("U", inplace=True)
    return data



y = train["Survived"]
X = train.drop("Survived", axis=1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)

predictions = clf.predict(X_val)
predictions2 = neigh.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)


submission_preds = clf.predict(test)
submission_preds2 = neigh.predict(test)


df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds,
                  })
df2 = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds2,
                  })
df.to_csv("submission.csv", index=False)
df2.to_csv("/kaggle/working/submission2.csv", index=False)
