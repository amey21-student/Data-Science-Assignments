#Data Exploration
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.impute import SimpleImputer

train = pd.read_csv("Titanic_train.csv")
test = pd.read_csv("Titanic_test.csv")

train.info()
train.describe()
train.head()
train.isnull().sum()
test.isnull().sum()

sns.histplot(train['Age'], kde=True)
sns.boxplot(x='Survived',y='Age', data=train)
sns.heatmap(train.select_dtypes(include='number').corr(),annot=True,cmap='coolwarm')

#Data Preprocessing
train['Age']=train['Age'].fillna(train['Age'].median())
test['Age']=test['Age'].fillna(test['Age'].median())

passenger_ids = test['PassengerId']

combined = pd.concat([train,test], sort=False)
combined = pd.get_dummies(combined,columns = ['Sex','Embarked'],drop_first=True)
combined = combined.drop(['Name','Ticket','Cabin',],axis=1,errors='ignore')
train = combined[:len(train)]
test = combined[len(train):]

X_train = train.drop(['Survived','SibSp','Parch','PassengerId'],axis=1,errors='ignore')
y_train = train['Survived']

X_test = test.drop(['Survived','SibSp','Parch','PassengerId'],axis=1,errors='ignore')

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(imputer.fit_transform(X_test),columns=X_test.columns)

X_tr,X_te,y_tr,y_te = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_tr,y_tr)
y_pred = model.predict(X_te)

print("\nAccuracy Score :",accuracy_score(y_te,y_pred))
print("\nPrecision Score :",precision_score(y_te,y_pred))
print("\nRecall Score :",recall_score(y_te,y_pred))
print("\nF1 Score :",f1_score(y_te,y_pred))

y_test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId' : passenger_ids,
    'Survived' : y_test_pred
})
submission.to_csv('titanic_submission.csv',index=False)

joblib.dump(model,"titanic_model.pkl")
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class (1=1st,2=2nd,3=3rd)",[1,2,3])
Age = st.number_input("Age",0,100,25)
Fare = st.number_input("Fare",0,600,100)
Sex_male = st.selectbox("Gender",["Male","Female"])
Embarked_Q = st.selectbox("Embarked Port (Q/S)", ["S", "Q"])
Embarked_S = 1 if Embarked_Q == "S" else 0
Sex_male = 1 if Sex_male == "Male" else 0


input_data = pd.DataFrame({
    'Pclass' : [Pclass],
    'Age' : [Age],
    'Fare' : [Fare],
    'Sex_male' : [Sex_male],
    'Embarked_Q' : [1 if Embarked_Q == "Q" else 0],
    'Embarked_S' : [Embarked_S]
})

if st.button("Predict Survival") :
    prediction = model.predict(input_data)
    st.success("Survived" if prediction == 1 else "Did not survive")

    