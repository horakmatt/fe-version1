import streamlit as st
import pandas as pd
import numpy as np
from latex import build_pdf
import io
import utils

from catboost import *
from catboost import datasets
import shap


# from pylatex import Document, Section, Command
# from pylatex.utils import NoEscape

df_ground = pd.read_csv('data/train.csv')
df_sample_data = pd.read_csv('data/sample_data.csv')
sample_data = df_sample_data.to_csv(index=False)
with open('latex_data/letter_template.tex', 'r') as f:
    base_letter = f.read()


model = CatBoostRegressor()
model.load_model('model_files/reg1.cbm')

st.title("Loan Application Evaluation Dashboard")

st.subheader("General Model Information.")

st.write(f"""This dashboard asks you to upload a loan applicant's loan information in a csv-format file containing only column headers and one line of data.  A sample applicant information csv file is available for download below.""")

st.write(f"""The underlying model calculates a decision score based on how this applicant's data fits into the data for recent historical loan decisions that the model was 'trained' on.  The model attempts to mimic those decisions as closely as possible for new applications.""")

st.write(f"""The score is a number between 0 and 1, with higher numbers indicating a more favorable decision towards 'accept'.""")
st.write(f"""A recommended loan decision is made based on where the applicant's score falls with respect to two pre-determined thresholds.""")
st.write(f"""The thresholds are given below, and they were also determined based on recent historical loan decisions.""")

st.write(f"""
Thresholds for decisions are:  decline/review threshold = 0.2, review/accept threshold = 0.8. 
In terms of zones for the decline/review/accept decision:\n
DECLINE_ZONE < 0.2 < REVIEW_ZONE < 0.8 < ACCEPT_ZONE\n""")

st.write("The data you enter should be in csv form and have the same format as sample data that can be downloaded below.")
st.download_button(
    label="Download Sample Data",
    data=sample_data,
    file_name="summary.csv",
    mime="text/csv",
)


st.subheader("Upload Loan application")
st.write("Please upload the loan application to be evaluated in csv format following the sample above.")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    # try:
        # Read the uploaded CSV file into a pandas DataFrame
    df_samp = pd.read_csv(uploaded_file)
    df_samp.reset_index(inplace=True)
    app_name = df_samp.loc[0,'application_id']
    st.write(f"We received and will evaluate the application data for applicant {app_name}.")
    df_samp = utils.load_and_process(df_samp)
    decision, prob = utils.eval_app(model, df_samp)
    shap_ser = utils.get_shap_values(model, df_samp)
    dict_pos, dict_neg = utils.get_reasons_for_lender(df_ground, shap_ser, df_samp=df_samp)
    explanation = utils.make_explanation_string(decision, prob, dict_pos, dict_neg, app_name)


    st.subheader("Loan Decision")
    decision_string = f"""The recommended decision for {app_name} is {decision.upper()} because the model score for this application is {prob:.4f}.  """
    decision_string = f"{decision_string}A more detailed explanation of the top reasons for this decision can be downloaded below in plain text format."""
    st.write(decision_string)
    # st.write(explanation)

    st.download_button(
        label="Download decision reasons",
        data=explanation,
        file_name="summary.txt",
        mime="text/csv",
    )

    if decision == "Decline":
        st.subheader("Letter to the applicant")
        st.write("Below is a letter to the applicant in pdf format briefly summarizing the top three reasons for a 'DECLINE'  decision.")

        v2 = dict_neg[0][1]
        v3 = dict_neg[0][0]
        v4 = dict_neg[1][1]
        v5 = dict_neg[1][0]
        v6 = dict_neg[2][1]
        v7 = dict_neg[2][0]

        # v1 = 'a'
        # v2 = 'b'
        # v3 = 'c'
        # v4 = 'd'
        # v5 = 'e'
        # v6 = 'f'
        # v7 = 'g'

        # v2 = v2.replace('_', '*')
        v3 = v3.replace('_', '\\_')
        # v4 = v4.replace('_', '*')
        v5 = v5.replace('_', '\\_')
        # v6 = v6.replace('_', '*')
        v7 = v7.replace('_', '\\_')

        pdf_bytes = utils.make_letter_pdf(base_letter = base_letter,
                                          v1 = app_name,
                                          v2 = v2,
                                          v3 = v3,
                                          v4 = v4,
                                          v5 = v5,
                                          v6 = v6,
                                          v7 = v7)

        st.download_button(
            label=f"Download {app_name} customer letter in PDF",
            data=pdf_bytes,
            file_name=f"{app_name}_decline_letter.pdf",
            mime="application/pdf",
        )




    # except Exception as e:
    #     st.error(f"Error reading CSV file: {e}")
else:
    st.info("Please upload a CSV file to get started.")