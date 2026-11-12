import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Bankruptcy Prevention',layout='wide')
st.markdown(
    """
    <style>
    /* Reduce padding at the top of the sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    /* Reduce vertical spacing between widgets in the sidebar */
    .stSelectbox {
        margin-bottom: -15px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Input Features")
options=[0.0,0.5,1.0]

ind_risk=st.sidebar.selectbox('Industrial Risk',options)
mgmt_risk=st.sidebar.selectbox('Management Risk',options)
fin_flex=st.sidebar.selectbox('Financial Risk',options)
credibility=st.sidebar.selectbox('Credibility',options)
competitiveness=st.sidebar.selectbox('Compititiveness',options)
op_risk=st.sidebar.selectbox('Operating Risk',options)

st.sidebar.markdown('---')
predict_btn=st.sidebar.button('Predict')

st.markdown("<h1 style style='text-align: center; border: 2px solid black; padding: 20px;'><br>Bankruptcy Prevention</h1>", unsafe_allow_html=True)
st.markdown("<br><br>",unsafe_allow_html=True)

input_dict={
    "Industrial Risk":[ind_risk],
    "Management Risk":[mgmt_risk],
    "Financial Flexibility":[fin_flex],
    "Credibility":[credibility],
    "Competitiveness":[competitiveness],
    "Operating Risk":[op_risk]
}

input_df=pd.DataFrame(input_dict)

st.table(input_df)
st.markdown("<br>",unsafe_allow_html=True)

if predict_btn:
        with open('final_bankruptcy_model.pkl','rb') as file:
            model=joblib.load(file)
            
        input_data=np.array([[ind_risk,mgmt_risk,fin_flex,credibility,competitiveness,op_risk]])
        
        prob=model.predict_proba(input_data)[0][1] 

        st.markdown("<div style='text-align: center;'>",unsafe_allow_html=True)
        
        st.markdown(f"### Predicted Probability of Bankruptcy: {prob*100:.2f}% ")    
        st.markdown("</div>",unsafe_allow_html=True)
