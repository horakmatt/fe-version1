import streamlit as st
import pandas as pd
import json
import numpy as np
from latex import build_pdf
import io
import zipfile
from catboost import *
from catboost import datasets
import shap

import utils

df_ground = pd.read_csv('data/train.csv')
df_sample_data = pd.read_csv('data/sample_data.csv')
sample_data = df_sample_data.to_csv(index=False) #Create the data sample for users to dlwnload.
with open('latex_data/document_template.tex', 'r') as f:
    base_latex = f.read()

with open('data/dict_r2f.json', 'r') as f:
    dict_r2f = json.load(f)

with open('data/dict_f2r.json', 'r') as f:
    dict_f2r = json.load(f)

with open('data/dict_r2explain.json', 'r') as f:
    dict_r2explain = json.load(f)


model = CatBoostRegressor()
model.load_model('model_files/reg1.cbm')

st.title("Loan Application Evaluation Dashboard")
st.subheader("General Model Information.")
st.write(f"""This dashboard asks you to upload csv-format file containing information on one or multiple applicant's \
loan information.  The first line of the csv file must be the names of the fields in the application. A sample input \
csv file is available for download below.""")

st.write(f"""The underlying model calculates a decision score for each applicant based on how the applicant's data fits into the distribution \
for recent historical loan decisions that the model was 'trained' on.  The model attempts to mimic those decisions as \
closely as possible for new applications.  The score for each application is a number between 0 and 1, with higher numbers indicating a more favorable decision \
towards 'accept'.""")

st.markdown("<p>A recommended loan decision is made based on where the applicant's score falls with respect to the two \
pre-determined thresholds, which are given below.  They were also determined based on recent historical loan \
decisions.  The Decline, Review and Accept zones defined by those thresholds are:"
"<p style='text-align: center;'>DECLINE_ZONE < 0.2 < REVIEW_ZONE < 0.8 < ACCEPT_ZONE", unsafe_allow_html=True)

st.markdown(f"Results are provided back in three formats.\n"
"1.  A csv containing all of the decisions together with the top three fields for each application influencing the decision \
in the positive direction (towards 'accept') and the top three fields for each application influencing the decision \
in the negative direction (towards 'decline').  For each field given, the applicant's value in that field is given in \
the corresponding 'value' column and for numerical fields, that value's percentile among the data on which the model \
was trained is in the corresponding 'percentile column.\n"
"2.  A text file containing further plain-English summaries of each decision, indended for use by the loan officer.\n"
"3.  If there are any 'Decline' decisions, a zipped file containing a pdf file with justifications for each 'Decline' \
decision is provided.  The pdfs is intended for the applicants.""")

st.write("The data you enter should be in csv form and have the same format as sample data that \
can be downloaded below.")
st.download_button(
    label="Download Sample Data",
    data=sample_data,
    file_name="sample.csv",
    mime="text/csv",
)

st.subheader("Upload Loan application")

triage_or_full = ' '
while len(triage_or_full) == 1:
    triage_or_full = st.selectbox(
        'Do you want to run the triage model or full model?',
        ('', 'Triage model', 'Full model')
    )
if len(triage_or_full) > 1:
    st.write(f"Thank you.  Proceeding with the {triage_or_full.lower()}.")

    st.write("Please upload the csv file containing the loan applications to be evaluated in csv format following the sample above.")
    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file is not None:
        # try:
            # Read the uploaded CSV file into a pandas DataFrame
        df_samp = pd.read_csv(uploaded_file)
        df_samp.reset_index(inplace=True)
        app_name = df_samp.loc[0,'application_id']
        st.write(f"We received and will evaluate the application data for {len(df_samp)} applicants.")

        df_res, explanations, zip_bytes = utils.process_apps(df_samp=df_samp,
                                                            model=model,
                                                            df_ground=df_ground,
                                                             dict_r2f=dict_r2f,
                                                             dict_r2explain=dict_r2explain,
                                                             dict_r2explain_positive=dict_r2explain_positive,
                                                            base_latex=base_latex)
        summary_csv = df_res.to_csv(index=False)

        st.subheader("Loan Decisions and Downloads")
        n_accept = len(df_res[df_res['decision'] == 'Accept'])
        n_review = len(df_res[df_res['decision'] == 'Review'])
        n_decline = len(df_res[df_res['decision'] == 'Decline'])

        st.markdown(f"We successflly processed {len(df_res)} applicants with the following decision distribution.\n"
        f"* {n_accept} 'Accept' decisions\n"
        f"* {n_review} 'Review' decisions\n"
        f"* {n_decline} 'Decline' decisions")

        # st.markdown(" * abc\n"
        #             "* def\n"
        #             "* ghi")

        st.write("Full summaries are available below.")

        st.download_button(
            label="Download decision summary csv file",
            data=summary_csv,
            file_name="summary.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download decision reasons",
            data=explanations,
            file_name="decision_summaries.txt",
            mime="text/csv",
        )


        if zip_bytes is not None:
            st.download_button(
                label=f"Download .zip file containing customer decline documents.",
                data=zip_bytes,
                file_name=f"decline_documents.zip",
                mime="application/zip",
            )

        else:
            st.write(f"There are no declined applications in the uploaded data.")




        # except Exception as e:
        #     st.error(f"Error reading CSV file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")