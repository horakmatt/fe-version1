import pandas as pd
import numpy as np
from latex import build_pdf
import io
import zipfile
import time



from catboost import *
from catboost import datasets
import shap

COLUMNS = ['intake_intake_channel', 'intake_new_customer', 'intake_lender_segment',
       'intake_region', 'intake_sector', 'intake_business_age_years',
       'intake_annual_revenue_gbp', 'intake_requested_loan_gbp',
       'intake_requested_term_months', 'intake_loan_purpose',
       'intake_personal_guarantee', 'intake_collateral_type',
       'intake_bureau_score', 'intake_ccj_count', 'intake_default_history',
       'intake_kyc_pass', 'intake_aml_risk', 'intake_pep_hit',
       'intake_sanctions_hit', 'intake_doc_completeness', 'enrich_dscr',
       'enrich_leverage_ratio', 'enrich_cashflow_volatility', 'enrich_ltv',
       'enrich_risk_score', 'enrich_triage_priority']
CAT_FEATURES = [0, 1, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 18]

NUMS = ['intake_business_age_years',
 'intake_annual_revenue_gbp',
 'intake_requested_loan_gbp',
 'intake_requested_term_months',
 'intake_bureau_score',
 'intake_ccj_count',
 'intake_doc_completeness',
 'enrich_dscr',
 'enrich_leverage_ratio',
 'enrich_cashflow_volatility',
 'enrich_ltv',
 'enrich_risk_score',
 'enrich_triage_priority',
 'label']

DROPCOLS = ['application_id', 'application_date', 'reason_primary', 'reason_secondary', 'reason_tertiary', 'expected_decision', 'label']


def create_in_memory_zip(file_data: dict) -> bytes:
    """
    Creates a zip archive in memory from a dictionary of filenames and data.

    Args:
        file_data: A dictionary where keys are filenames and values are the
                   byte-like content of the files.

    Returns:
        A bytes object representing the complete zip archive.
    """
    # Create an in-memory binary stream
    in_memory_zip = io.BytesIO()

    # Open the stream as a zip file in write mode ('w')
    # The 'with' statement ensures the file is properly closed
    with zipfile.ZipFile(in_memory_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, data in file_data.items():
            # Use writestr() to add files to the archive from a string or bytes
            zipf.writestr(filename, data)

    # Return the bytes content of the in-memory buffer
    return in_memory_zip.getvalue()

def encode_label(s):
    if s == 'Accept':
        return 1.0
    elif s == 'Decline':
        return 0.0
    return 1/2


def load_and_process(application):
    columns = COLUMNS
    cat_features = CAT_FEATURES
    nums = NUMS
    dropcols = DROPCOLS

    if isinstance(application, pd.DataFrame):
        df = application.copy()
    else:
        df = pd.read_csv(application)
    for c in dropcols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    # try:
    df = df[columns]
    # except:
    #     print('Missing columns')

    for c in df.columns:
        if not c in nums:
            df[c] = df[c].fillna(value='None')

    return df


def classify(x, thresh_r, thresh_a):
    if x < thresh_r:
        return 0
    elif x < thresh_a:
        return 1/2
    return 1

def make_decision(prob, thresh_decline = 0.2, thresh_accept = 0.8):
    if prob < thresh_decline:
        return 'Decline'
    elif prob < thresh_accept:
        return 'Review'
    return 'Accept'

def eval_app(model, df_samp):
    samp = df_samp.iloc[0]
    prob = model.predict(data=samp)
    decision = make_decision(prob)
    return decision, prob

def get_shap_values(model, df_samp):
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer(df_samp)
    shap_ser = pd.Series(index=COLUMNS, data=shap_values.values.reshape(-1,))
    return shap_ser

def find_loc_in_data(df, col, x):
    n_less = df[df[col] <= x].shape[0]
    frac_less = n_less / len(df)
    return np.round(frac_less, 2)


def get_reasons_for_lender(df_ground, shap_ser, df_samp):
    ser_samp = df_samp.iloc[0]
    shap_pos = shap_ser[shap_ser>0].copy()
    npos = min(3, len(shap_pos))
    shap_pos.sort_values(ascending=False, inplace=True)
    pos = shap_pos.index[:npos]

    shap_neg = shap_ser[shap_ser<0].copy()
    nneg = min(3, len(shap_neg))
    shap_neg.sort_values(ascending=True, inplace=True)
    neg = shap_neg.index[:nneg]

    dict_pos = dict()
    for i,c in enumerate(pos):
        sv = shap_pos[c]
        val = ser_samp[c]
        if c in NUMS:
            position = find_loc_in_data(df_ground, c, val)
            dict_pos[i] = (c, val, position)
        else:
            dict_pos[i] = (c, sv, np.nan)
    dict_neg = dict()
    for i,c in enumerate(neg):
        sv = shap_neg[c]
        val = ser_samp[c]
        if c in NUMS:
            position = find_loc_in_data(df_ground, c, val)
            position = position
            dict_neg[i] = (c, val, position)
        else:
            dict_neg[i] = (c, val, np.nan)
    return dict_pos, dict_neg

def make_explanation_string(decision, prob, dict_pos, dict_neg, app_name):
    explanation = ''
    list_line_pos = []
    list_line_neg = []
    if decision == 'Accept':
        explanation = f"The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is above the accept threshold of accept_thresh=0.8.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following  If needed, a review could be started focusing on these."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is favorable in the ground truth data."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in still the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is not sufficiently unfavorable to influence a negative decision."
            list_line_neg.append(newline)

    elif decision == 'Decline':
        explanation = f"The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is below the decline threshold of decline_thresh = 0.2.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following.  If needed, a review could be started focusing on these."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is not sufficiently favorable in the ground truth data to influence a positive decision."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in only the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is unfavorable in the ground truth data."
            list_line_neg.append(newline)

    if decision == 'Review':
        explanation = f"The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is between the decline and accept thresholds of decline_thresh=0.2 and accept_thresh=0.8."
        explanation = f"{explanation}  Following are both positive and negative factors of the application that the model found.  A review could start by looking into these.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"Attribute {v[0]} of this application has value {v[1]}, which is in the top {v[2]}% of the data."
            else:
                newline = f"Attribute {v[0]} of this application has value {v[1]}."
            list_line_neg.append(newline)

    explanation = f"{explanation}{l1}"
    for line in list_line_pos:
        explanation += f"{line}\n"
    explanation = explanation + '\n'
    explanation = explanation + f"{l2}\n"
    for line in list_line_neg:
        explanation += f"{line}\n"

    return explanation

def make_letter_pdf(base_latex, app_name, dict_neg, dict_pos):
    nf1 = dict_neg[0][0]
    nv1 = dict_neg[0][1]
    nf2 = dict_neg[1][0]
    nv2 = dict_neg[1][1]
    nf3 = dict_neg[2][0]
    nv3 = dict_neg[2][1]
    pf1 = dict_pos[0][0]
    pv1 = dict_pos[0][1]

    nf1 = nf1.replace('_', '\\_')
    nf2 = nf2.replace('_', '\\_')
    nf3 = nf3.replace('_', '\\_')
    pf1 = pf1.replace('_', '\\_')


    base_latex = base_latex.replace('RESERVEDAPPID1', str(app_name))
    base_latex = base_latex.replace('RESERVEDNEGVALUE1', str(nv1))
    base_latex = base_latex.replace('RESERVEDNEGVALUE2', str(nv2))
    base_latex = base_latex.replace('RESERVEDNEGVALUE3', str(nv3))
    base_latex = base_latex.replace('RESERVEDNEGFIELD1', str(nf1))
    base_latex = base_latex.replace('RESERVEDNEGFIELD2', str(nf2))
    base_latex = base_latex.replace('RESERVEDNEGFIELD3', str(nf3))
    base_latex = base_latex.replace('RESERVEDPOSVALUE1', str(pv1))
    base_latex = base_latex.replace('RESERVEDPOSFIELD1', str(pf1))

    pdf_object = build_pdf(base_latex)
    pdf_bytes = pdf_object.readb()

    return pdf_bytes

def process_apps(df_samp, model, df_ground, base_latex):
    # df_samp.reset_index(inplace=True)
    cols = ['application_id', 'decision', 'score',
            'pos_field1', 'pos_value1', 'pos_percentile1',
            'pos_field2', 'pos_value2', 'pos_percentile2',
            'pos_field3', 'pos_value3', 'pos_percentile3',
            'neg_field1', 'neg_value1', 'neg_percentile1',
            'neg_field2', 'neg_value2', 'neg_percentile2',
            'neg_field3', 'neg_value3', 'neg_percentile3']
    rows = []
    explanations = ''
    dict_zip = dict()
    for i in df_samp.index:
        print(f"{pd.to_datetime(time.time(), unit='s')}, Processing {i}")
        df_app = df_samp.loc[[i]]
        row, explanation, pdf_bytes = process_one(df_app, model, df_ground, base_latex)
        rows.append(row)
        explanations = f"{explanations}\n\n{explanation}"
        if pdf_bytes is not None:
            dict_zip[f"{row[0]}.pdf"] = pdf_bytes
    df_res = pd.DataFrame(data=rows, columns=cols)
    if len(dict_zip) > 0:
        zip_bytes = create_in_memory_zip(dict_zip)
    else:
        zip_bytes = None


    return df_res, explanations, zip_bytes




def process_one(df_app, model, df_ground, base_latex):
    df_app.reset_index(inplace=True)
    app_name = df_app.loc[0,'application_id']
    df_app = load_and_process(df_app)
    decision, prob = eval_app(model, df_app)
    shap_ser = get_shap_values(model, df_app)
    dict_pos, dict_neg = get_reasons_for_lender(df_ground, shap_ser, df_samp=df_app)
    explanation = make_explanation_string(decision, prob, dict_pos, dict_neg, app_name)
    row = [app_name, decision, prob]
    for i in range(3):
        if i in dict_pos.keys():
            row.extend(dict_pos[i])
        else:
            row.extend(['None', np.nan, np.nan])
    for i in range(3):
        if i in dict_neg.keys():
            row.extend(dict_neg[i])
        else:
            row.extend(['None', np.nan, np.nan])

    if decision == 'Decline':
        pdf_bytes = make_letter_pdf(base_latex=base_latex,
                                    app_name=app_name,
                                    dict_neg=dict_neg,
                                    dict_pos=dict_pos)
    else:
        pdf_bytes = None

    return row, explanation, pdf_bytes





