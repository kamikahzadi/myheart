import numpy as np 
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
st.write("""
# Simple Heart Disease Prediction App
This app predicts the **HeartDisease** !
""")

df = pd.read_csv('https://github.com//kamikahzadi//myheart//blob//main//data//heart.csv')
df['Sex'] = df['Sex'].map({'M':1 , 'F':0})
df['ChestPainType'] = df['ChestPainType'].map({'ATA':1, 'NAP':2, 'ASY':3, 'TA':0})
df['RestingECG'] = df['RestingECG'].map({'Normal':1, 'ST':0, 'LVH':2})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N':0, 'Y':1})
df['ST_Slope'] = df['ST_Slope'].map({'Up':1, 'Flat':0, 'Down':2})


def users():
    age = st.number_input('enter the age')
    sex = st.radio('your SEX' , [0,1])
    chpt = st.slider('chestpaintype' ,0,3 )
    resecg = st.radio('RestECG' , [0 , 1 , 2])
    excerang = st.number_input('enter it 1 or 0')
    stsl = st.slider('upflattdown' , 0 ,1,2)
    resb = st.number_input('your resb')
    chol = st.number_input('clolestrol')
    fas = st.radio('fes' , [0,1])
    mx = st.number_input('its your MAXH')
    old = st.slider('old' , 0,6)
    data = {'age':age ,'SEX':sex , 'ChestPainType':chpt ,
    'resbp':resb ,'Cholostrol':chol ,'fasbs':fas,'RestingECG':resecg ,'Maxhr':mx ,'ExerciseAngina':excerang,
    'oldpeak':old,'ST_Slope':stsl , }
    features = pd.DataFrame(data , index=[0])
    return features

d = users()

x = df.iloc[: , :-1].values
y = df.iloc[: , -1].values
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25 , random_state=42)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

rf = RandomForestClassifier(n_estimators=300)
rf.fit(x_train , y_train)
prediction = rf.predict(d)
y_pred = rf.predict(x_test)

st.subheader('your features')
st.write(d)

st.subheader('The feedback of your Heart Disease quiery')
st.write('result is',prediction)

pred_prob = rf.predict_proba(d)

st.subheader('probability')
st.write(pred_prob)

accuracy = accuracy_score(y_test,y_pred)*100
st.write('the accuracy of prediction is {:.2f} %'.format(accuracy))



#
if st.button('anything else?') == True:
     st.success('nothing you just got it')
     st.balloons()
