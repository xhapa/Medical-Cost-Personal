import streamlit as st

import numpy as np 
import pandas as pd 

import pickle

text = '''# Medical Cost Personal Prediction 🚑️
This app predict the **charges** of insurance
'''

st.write(text)

with st.sidebar:
    header = st.header('User input features')
    st.markdown('[Example CSV input file]()')
    uploaded_file = st.file_uploader('Upload your input CSV file', type=['csv'])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            age = st.slider('Age', 18, 64, 25)
            sex= st.selectbox('Sex', ('male', 'female'))
            bmi = st.slider('BMI', 15.96, 53.13, 25.0)
            children = st.selectbox('N° Children', (0, 1, 2, 3, 4, 5))
            smoker = st.selectbox('Smoker', ('yes', 'no'))
            region = st.selectbox('Region', ('northeast', 'southeast', 'southwest', 'northwest'))
            data = {
                    'age':age, 
                    'sex':sex,
                    'bmi':bmi,
                    'children':children,
                    'smoker':smoker,
                    'region':region
                    }
            features = pd.DataFrame(data, index=[0])
            return features
        
        input_df = user_input_features()

medical_cost_df = pd.read_csv('insurance.csv')
medical_cost_df = medical_cost_df.drop(columns=['charges'])

df = pd.concat([input_df, medical_cost_df], axis=0)
df_to_model = pd.get_dummies(df, ['sex', 'smoker', 'region'], drop_first=True)

df_to_model['age2'] = df_to_model.age**2
df_to_model['overweight'] = (df_to_model['bmi'] >= 30).astype(int)
df_to_model['owbysmok'] = df_to_model.smoker_yes*df_to_model.overweight


user_data = df[:1]

st.subheader('User input features')

if uploaded_file is not None:
    st.write(user_data)

else: 
    st.write('Awaiting CSV file to be uploading. Currently using example input parameters.')
    st.dataframe(user_data)

load_model = pickle.load(open('./models/medical_cost_mod.pkl', 'rb'))
load_sc_X = pickle.load(open('./models/scalerX.pkl', 'rb'))
load_sc_Y = pickle.load(open('./models/scalerY.pkl', 'rb'))

model_features = ['age2', 'owbysmok', 'smoker_yes', 'children']

st.subheader('Prediction in $USD')

if uploaded_file is not None:
    X_std = load_sc_X.transform(df_to_model[model_features].values)
    Y_pred_std = load_model.predict(X_std)
    Y_pred = load_sc_Y.inverse_transform(Y_pred_std)
    Y_pred_df = pd.DataFrame(Y_pred)
    output_df = pd.concat([input_df, Y_pred_df], axis=1)
    st.write(output_df[:5])
else:
    X_std = load_sc_X.transform(df_to_model[:1][model_features].values)
    Y_pred_std = load_model.predict(X_std)
    Y_pred = load_sc_Y.inverse_transform(Y_pred_std)
    st.write(Y_pred)