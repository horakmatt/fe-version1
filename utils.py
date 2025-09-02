import pandas as pd
import numpy as np
from latex import build_pdf

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
    n_less = df[df[col] < x].shape[0]
    frac_less = n_less / len(df)
    return np.round(100*frac_less, 2)


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

def make_letter_pdf(base_letter, v1, v2, v3, v4, v5, v6, v7):
    base_letter = base_letter.replace('RESERVEDSTRING1', str(v1))
    base_letter = base_letter.replace('RESERVEDSTRING2', str(v2))
    base_letter = base_letter.replace('RESERVEDSTRING3', str(v3))
    base_letter = base_letter.replace('RESERVEDSTRING4', str(v4))
    base_letter = base_letter.replace('RESERVEDSTRING5', str(v5))
    base_letter = base_letter.replace('RESERVEDSTRING6', str(v6))
    base_letter = base_letter.replace('RESERVEDSTRING7', str(v7))

    pdf_object = build_pdf(base_letter)
    pdf_bytes = pdf_object.readb()

    return pdf_bytes







