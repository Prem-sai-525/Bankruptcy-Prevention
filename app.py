import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Bankruptcy Prediction', layout='wide')

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.title("📉 Bankruptcy Prediction")
st.write("Predict whether a company is at risk of bankruptcy from six risk factors. Each factor is rated Low, Medium or High.")

# Create Tabs
tab1, tab2, tab_eda, tab3 = st.tabs(["Single company", "Batch prediction (CSV)", "EDA", "About"])

risk_mapping = {"0.0 (Low)": 0.0, "0.5 (Medium)": 0.5, "1.0 (High)": 1.0}
slider_options = ["0.0 (Low)", "0.5 (Medium)", "1.0 (High)"]

with tab1:
    st.subheader("Enter the company's risk factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ind_risk_label = st.select_slider('Industrial Risk', options=slider_options, value="0.0 (Low)")
        fin_flex_label = st.select_slider('Financial Flexibility', options=slider_options, value="0.5 (Medium)")
        comp_label = st.select_slider('Competitiveness', options=slider_options, value="0.5 (Medium)")
        
    with col2:
        mgmt_risk_label = st.select_slider('Management Risk', options=slider_options, value="0.5 (Medium)")
        cred_label = st.select_slider('Credibility', options=slider_options, value="0.5 (Medium)")
        op_risk_label = st.select_slider('Operating Risk', options=slider_options, value="0.5 (Medium)")

    st.markdown("<br>", unsafe_allow_html=True)
    

    left_col, center_col, right_col = st.columns([1, 1, 1])
    
    with center_col:
        predict_btn = st.button('Predict', type='primary', use_container_width=True)

    if predict_btn:
        ind_risk = risk_mapping[ind_risk_label]
        mgmt_risk = risk_mapping[mgmt_risk_label]
        fin_flex = risk_mapping[fin_flex_label]
        credibility = risk_mapping[cred_label]
        competitiveness = risk_mapping[comp_label]
        op_risk = risk_mapping[op_risk_label]

        try:
            with open('final_bankruptcy_model.pkl', 'rb') as file:
                model = joblib.load(file)
                
            input_data = np.array([[ind_risk, mgmt_risk, fin_flex, credibility, competitiveness, op_risk]])
            prob = model.predict_proba(input_data)[0][1] 

            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(f"### Predicted Probability of Bankruptcy: {prob*100:.2f}% ")    
            st.markdown("</div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Model file 'final_bankruptcy_model.pkl' not found.")

with tab2:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    left_col2, center_col2, right_col2 = st.columns([1, 1, 1])
    
    with center_col2:
        predict_batch_btn = st.button('Predict Batch', type='primary', use_container_width=True, key="batch_predict")
    
    if predict_batch_btn:
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                

                with open('final_bankruptcy_model.pkl', 'rb') as file:
                    model = joblib.load(file)
                
                probabilities = model.predict_proba(input_df)[:, 1]
                predictions = model.predict(input_df)
                
                results_df = input_df.copy()
                results_df['Probability of Bankruptcy (%)'] = (probabilities * 100).round(2)
                results_df['Prediction'] = ['Bankruptcy Risk' if p == 1 else 'Safe' for p in predictions]
                
                st.success("Batch prediction completed successfully!")
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                
                dl_left, dl_center, dl_right = st.columns([1, 1, 1])
                with dl_center:
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='batch_predictions_results.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a CSV file first.")

with tab_eda:
    st.subheader("Exploratory Data Analysis")

    @st.cache_data
    def load_eda_data(path):
        d = pd.read_csv(path, sep=';', skipinitialspace=True)
        d.columns = d.columns.str.strip()
        return d

    try:
        eda_df = load_eda_data('Bankruptcy_prevention_Dataset.csv')
    except FileNotFoundError:
        st.warning("Dataset 'Bankruptcy_prevention_Dataset.csv' not found. Upload it to explore.")
        eda_up = st.file_uploader("Upload the dataset (semicolon-separated CSV)", type=["csv"], key="eda_upload")
        eda_df = None
        if eda_up is not None:
            eda_df = pd.read_csv(eda_up, sep=';', skipinitialspace=True)
            eda_df.columns = eda_df.columns.str.strip()

    if eda_df is not None:
        feature_cols = [c for c in eda_df.columns if c != 'class']
        pretty = lambda c: c.replace('_', ' ').title()

        # Overview metrics
        counts = eda_df['class'].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Companies", len(eda_df))
        m2.metric("Features", len(feature_cols))
        m3.metric("Bankruptcy", int(counts.get('bankruptcy', 0)))
        m4.metric("Non-bankruptcy", int(counts.get('non-bankruptcy', 0)))

        with st.expander("Preview data & summary statistics", expanded=False):
            st.dataframe(eda_df.head(20), use_container_width=True)
            st.dataframe(eda_df[feature_cols].describe().T, use_container_width=True)
            st.caption(f"Missing values: {int(eda_df.isna().sum().sum())} | Duplicate rows: {int(eda_df.duplicated().sum())}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=eda_df, x='class', hue='class', legend=False, ax=ax,
                          palette={'bankruptcy': '#d9534f', 'non-bankruptcy': '#5cb85c'}, order=['bankruptcy', 'non-bankruptcy'])
            for cont in ax.containers:
                ax.bar_label(cont)
            ax.set_xlabel("")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.markdown("**Correlation heatmap**")
            corr_df = eda_df[feature_cols].copy()
            corr_df['class (bankruptcy=1)'] = (eda_df['class'] == 'bankruptcy').astype(int)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Feature distributions**")
        view = st.radio("View", ["Overall", "By class"], horizontal=True, key="eda_view")
        dcols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with dcols[i % 3]:
                fig, ax = plt.subplots(figsize=(4, 3))
                if view == "Overall":
                    sns.histplot(eda_df, x=col, bins=5, ax=ax, color='#4c78a8')
                else:
                    sns.histplot(eda_df, x=col, hue='class', multiple='dodge', bins=5, ax=ax,
                                 palette={'bankruptcy': '#d9534f', 'non-bankruptcy': '#5cb85c'})
                ax.set_title(pretty(col))
                ax.set_xlabel("")
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("**Boxplot of all features**")
        fig, ax = plt.subplots(figsize=(11, 4))
        eda_df[feature_cols].boxplot(ax=ax)
        ax.set_xticklabels([pretty(c) for c in feature_cols])
        st.pyplot(fig)
        plt.close(fig)


with tab3:
    st.subheader("Project Objective")
    st.write("This is a classification project aimed at modeling the probability that a business goes bankrupt based on different features, predicting a binary outcome of bankruptcy or non-bankruptcy.")
    
    st.subheader("Feature Details")
    st.markdown("""
    * **Industrial Risk:** 0 = low risk, 0.5 = medium risk, 1 = high risk
    * **Management Risk:** 0 = low risk, 0.5 = medium risk, 1 = high risk
    * **Financial Flexibility:** 0 = low flexibility, 0.5 = medium flexibility, 1 = high flexibility
    * **Credibility:** 0 = low credibility, 0.5 = medium credibility, 1 = high credibility
    * **Competitiveness:** 0 = low competitiveness, 0.5 = medium competitiveness, 1 = high competitiveness
    * **Operating Risk:** 0 = low risk, 0.5 = medium risk, 1 = high risk
    * **Class:** bankruptcy, non-bankruptcy (target variable)
    """)