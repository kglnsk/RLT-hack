import streamlit as st
import pandas as pd
import numpy as np
import shap
import catboost as cb

data = pd.read_csv('data.csv')

st.sidebar.title('Upload your data')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

model = cb.CatBoostClassifier()
model.load_model('rlt-hack')

st.write(data)

if uploaded_file is not None:
    data = pd.read_csv('data.csv')
with st.form(key='my_form'):
    st.write(data.style.highlight_null(null_color='red'))
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        data = pd.read_csv(uploaded_file)
        edited_df = st.experimental_data_editor(data)



if submit_button:
    data = edited_df
    predictions = model.predict(data)
    predictions_proba = model.predict_proba(data)
    predictions_df = pd.DataFrame(predictions_proba, columns=model.classes_)
    predictions_df['Predictions'] = predictions
    predictions_df = predictions_df[['Predictions'] + list(model.classes_)]
    st.write(predictions_df.style.apply(lambda x: ['background: green' if i==1 else '' for i in x], axis=1))
   
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, data)



