import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.write("""
    # Diabetes Dedection
    Detect if someone has diabetes using ML
""")

df = pd.read_csv('diabetes.csv')
st.subheader('Data Information')
st.write(df.describe())
chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('bmi', 0, 67, 32)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81,29)

    user_data = {
        'Pregnancies' : Pregnancies,
        'Glucose' : Glucose,
        'BloodPressure' : BloodPressure,
        'SkinThickness' : SkinThickness,
        'Insulin' : Insulin,
        'bmi' : bmi,
        'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
        'Age' : Age,
    }

    features = pd.DataFrame(user_data, index = [0])
    return features


user_input = get_user_input()
st.subheader('User Input:')

# model
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
st.subheader('All Time Model Test Accuracy:')
st.write(str(accuracy_score(y_test, rfc.predict(x_test))* 100) + '%')

prediction = rfc.predict(user_input)
st.subheader('Your Classification Result:')
st.write(prediction)