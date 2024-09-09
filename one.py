import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
from sklearn import tree

df = pd.read_csv('train.csv')
median_age = math.floor(df.Age.median())
df.Age = df.Age.fillna(median_age)
inputs = df.drop('Survived', axis='columns')
traget = df['Survived']

st.sidebar.header('USER INPUTS')

user_input = {
    'PassengerId': st.sidebar.number_input('PassengerId',min_value=0,step=1),
    'Pclass': st.sidebar.number_input('Pclass',min_value=0,step=1),
    'Age': st.sidebar.number_input('Age',min_value=0.00,step=0.01),
    'SibSp': st.sidebar.number_input('SibSp',min_value=0,step=1),
    'Parch':st.sidebar.number_input('Parch',min_value=0,step=1),
    'Fare':st.sidebar.number_input('Fare', min_value=0.0, step=0.0001),
    'SEX':st.sidebar.number_input('SEX',min_value=0,step=1,max_value=1),
    'EMBARKED':st.sidebar.number_input('EMBARKED',min_value=0,step=1)
}

input_features = np.array([[
    user_input['PassengerId'],
    user_input['Pclass'],
    user_input['Age'],
    user_input['SibSp'],
    user_input['Parch'],
    user_input['Fare'],
    user_input['SEX'],
    user_input['EMBARKED']
]])

le_sex = LabelEncoder()
inputs['SEX'] = le_sex.fit_transform(inputs['Sex'])
inputs['EMBARKED'] = le_sex.fit_transform(inputs['Embarked'])
inputs = inputs.drop(['Sex', 'Embarked'], axis='columns')
st.write(inputs)

model = tree.DecisionTreeClassifier()
model.fit(inputs, traget)
X = model.score(inputs, traget)
st.write(X)

Y = model.predict(input_features)

st.write(Y)
if Y == 0:
    st.write('DIDNT SURVIVE')
else:
    st.write('SURVIVED')

st.write(traget)