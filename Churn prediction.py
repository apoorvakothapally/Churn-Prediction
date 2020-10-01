import pandas as pd
import os
import matplotlib as plt
%matplotlib inline
df = pd.read_csv("Downloads/churn_prediction.csv")
df.isnull().sum()
df.dtypes
df['gender'].fillna(value=df['gender'].mode()[0],inplace = True)
df['occupation'].fillna(value=df['occupation'].mode()[0], inplace = True)
df['dependents'].fillna(value=df['dependents'].mean(), inplace = True)
df['days_since_last_transaction'].fillna(value=df['days_since_last_transaction'].mean(), inplace = True)
df.drop(columns=['city','customer_nw_category','current_month_balance','previous_month_end_balance'],axis=1,inplace = True)
df
df.describe()
df.isnull().sum()
df['gender'] = df['gender'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
import matplotlib.pyplot as plt
%matplotlib inline
i=0
for i in range(0,len(df.columns)):
    df.plot.scatter(i,16)
df[df<0] = 0
df.describe()
from sklearn import preprocessing
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df=pd.DataFrame(x_scaled)
df
import matplotlib.pyplot as plt
%matplotlib inline
i=0
for i in range(0,len(df.columns)):
    df.plot.scatter(i,16)
x=df.drop([16],axis=1)
y=df[16]
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=90,stratify=y)
test_x
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
y_predict = clf.predict(test_x)
print("accuracy:",metrics.accuracy_score(test_y,y_predict))
user_input = [None]*16
for i in range(0,len(test_x.columns)):
    user_input[i] = input()
user_input = pd.DataFrame([user_input])
frame = [test_x,user_input]
test_x = pd.concat(frame)
print(test_x)
test_x = min_max_scaler.fit_transform(test_x)
y_pre = clf.predict(test_x)
print(y_pre[-1])