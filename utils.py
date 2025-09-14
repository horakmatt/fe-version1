import pandas as pd
import numpy as np
from latex import build_pdf
import io
import zipfile
import time



from catboost import *
from catboost import datasets
import shap
from numba.core.types import np_uint64

THRESH_DECLINE = 0.2
THRESH_ACCEPT = 0.8
THRESH_TRIAGE_DECLINE = 0.5

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

COLUMNS_TRIAGE = ['intake_intake_channel', 'intake_new_customer', 'intake_lender_segment',
       'intake_region', 'intake_sector', 'intake_business_age_years',
       'intake_annual_revenue_gbp', 'intake_requested_loan_gbp',
       'intake_requested_term_months', 'intake_loan_purpose',
       'intake_personal_guarantee', 'intake_collateral_type',
       'intake_bureau_score', 'intake_ccj_count', 'intake_default_history',
       'intake_kyc_pass', 'intake_aml_risk', 'intake_pep_hit',
       'intake_sanctions_hit', 'intake_doc_completeness']
CAT_FEATURES_TRIAGE = [0, 1, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 18]

NUMS_TRIAGE = ['intake_business_age_years',
 'intake_annual_revenue_gbp',
 'intake_requested_loan_gbp',
 'intake_requested_term_months',
 'intake_bureau_score',
 'intake_ccj_count',
 'intake_doc_completeness',
 'label']

DROPCOLS = ['application_id', 'application_date', 'reason_primary', 'reason_secondary', 'reason_tertiary', 'expected_decision', 'label']

DUPLICATE_REASONS = ['REVIEW_SECURITY',
                     'REVIEW_CREDIT_HISTORY',
                     'REVIEW_AFFORDABILITY',
                     'KYC_FAIL',
                     'REVIEW_COMPLIANCE']

HARD_STOPS = ['intake_kyc_pass',
              'intake_pep_hit',
              'intake_sanctions_hit']

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

def encode_label_triage(s):
    if s == 'Accept':
        return 1.0
    return 0.0


def load_and_process(application, tf = 'full'):
    if tf == 'full':
        columns = COLUMNS
        cat_features = CAT_FEATURES
        nums = NUMS
    else:
        columns = COLUMNS_TRIAGE
        cat_features = CAT_FEATURES_TRIAGE
        nums = NUMS_TRIAGE
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


def classify_traige(x, thresh_r):
    if x < thresh_r:
        return 0
    return 1

def make_decision(prob, thresh_decline = None, thresh_accept = None, thresh_triage_decline = None, tf = 'full'):
    if not thresh_decline:
        thresh_decline = THRESH_DECLINE
    if not thresh_accept:
        thresh_accept = THRESH_ACCEPT
    if not thresh_triage_decline:
        thresh_triage_decline = THRESH_TRIAGE_DECLINE

    if tf == 'full':
        if prob < thresh_decline:
            return 'Decline'
        elif prob < thresh_accept:
            return 'Review'
        return 'Accept'
    else:
        if prob < thresh_triage_decline:
            return 'Decline'
        else:
            return 'Do Not Decline'

def eval_app(model, df_samp, tf = 'full'):
    """

    :param model:
    :param df_samp:
    :param tf:  string telling which model to use.  'full' = use full model, 'triage' = use triage model.
    :return:
    """
    samp = df_samp.iloc[0]
    prob = model.predict(data=samp)
    decision = make_decision(prob, tf=tf)
    return decision, prob

def get_shap_values(model, df_samp, tf = 'full'):
    if tf == 'full':
        columns = COLUMNS
    else:
        columns = COLUMNS_TRIAGE
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer(df_samp)
    shap_ser = pd.Series(index=columns, data=shap_values.values.reshape(-1,))
    return shap_ser

def find_loc_in_data(df, col, x):
    df1 = df[[col]].copy()
    df1 = df1.fillna(value=0)
    n_less = df1[df1[col] <= x].shape[0]
    frac_less = n_less / len(df1)
    return np.round(frac_less, 2)


def get_reasons_for_lender(df_ground, shap_ser, df_samp):
    ser_samp = df_samp.iloc[0]
    shap_neg = shap_ser[shap_ser<0].copy()
    nneg = min(3, len(shap_neg))
    shap_neg.sort_values(ascending=True, inplace=True)
    neg = shap_neg.index[:nneg]

    shap_pos = shap_ser[shap_ser>0].copy()
    npos = min(3, len(shap_pos))
    if npos > 0:
        shap_pos.sort_values(ascending=False, inplace=True)
        pos = shap_pos.index[:npos]
    else: #If there are no positive reasons, take the least negative one.
        pos = shap_neg.index[-1:]

    dict_pos = dict()
    for i,c in enumerate(pos):
        sv = shap_pos[c]
        val = ser_samp[c]
        if c in NUMS:
            pctile = find_loc_in_data(df_ground, c, val)
            dict_pos[i] = (c, val, pctile)
        else:
            dict_pos[i] = (c, sv, np.nan)
    dict_neg = dict()
    for i,c in enumerate(neg):
        sv = shap_neg[c]
        val = ser_samp[c]
        if c in NUMS:
            pctile = find_loc_in_data(df_ground, c, val)
            dict_neg[i] = (c, val, pctile)
        else:
            dict_neg[i] = (c, val, np.nan)
    dict_neg = promote_hard_stops(dict_neg_orig=dict_neg)
    return dict_pos, dict_neg

def make_explanation_string(decision, prob, dict_pos, dict_neg, app_name, appnum = None, tf = 'full'):
    """

    :param decision:
    :param prob:
    :param dict_pos:
    :param dict_neg:
    :param app_name:
    :param appnum: Optional number to be used in the text file to the lender to identify the row in the input data
        for this application.
    :return:
    """
    if isinstance(decision, int):
        explanation = f"\n({appnum+1}) {app_name}\n"
    else:
        explanation = f"\n{app_name}\n"
    list_line_pos = []
    list_line_neg = []
    if decision == 'Accept':
        explanation = f"{explanation}The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is above the accept threshold of accept_thresh = {THRESH_ACCEPT}.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following  If needed, a review could be started focusing on these."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}. Attribute {v[0]} of this application has value {v[1]}, which is favorable in the ground truth data."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in still in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is not sufficiently unfavorable to influence a negative decision."
            list_line_neg.append(newline)

    elif decision == 'Decline':
        if tf == 'full':
            explanation = f"{explanation}The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is below the decline threshold of decline_thresh = {THRESH_DECLINE}.\n\n"
        else:
            explanation = f"{explanation}The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is below the triage 'do not decline' threshold of triage_decline_thresh = {THRESH_TRIAGE_DECLINE}.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following.  If needed, a review could be started focusing on these."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is not sufficiently favorable in the ground truth data to influence a positive decision."
            list_line_pos.append(newline)

        # print(app_name, len(dict_neg))
        for i in range(len(dict_neg)):
            v = dict_neg[i]
            # print(i,v)
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in only the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is unfavorable in the ground truth data."
            list_line_neg.append(newline)

    if decision == 'Review':
        explanation = f"{explanation}The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is between the decline and accept thresholds of decline_thresh = {THRESH_DECLINE} and accept_thresh = {THRESH_ACCEPT}."
        explanation = f"{explanation}  Following are both positive and negative factors of the application that the model found.  A review could start by looking into these.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}."
            list_line_neg.append(newline)

    if decision == 'Do Not Decline':
        explanation = f"{explanation}The model recommends {decision.upper()} for {app_name} because the model score of {prob:.4f} is above the triage 'do not decline' threshold of triage_decline_thresh = {THRESH_TRIAGE_DECLINE}."
        explanation = f"{explanation}  Following are both positive and negative factors of the application that the model found.  A review could start by looking into these.\n\n"
        l1 = "The top 3 positive contributing factors to this decision are following.\n"
        l2 = "The top 3 negative contributing factors to this decision are following."
        for i in range(len(dict_pos)):
            v = dict_pos[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}."
            list_line_pos.append(newline)

        for i in range(len(dict_neg)):
            v = dict_neg[i]
            if not np.isnan(v[2]):
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}, which is in the {100*v[2]:.2f} percentile of the data."
            else:
                newline = f"\t {i+1}.  Attribute {v[0]} of this application has value {v[1]}."
            list_line_neg.append(newline)



    explanation = f"{explanation}{l1}"
    for line in list_line_pos:
        explanation += f"{line}\n"
    explanation = explanation + '\n'
    explanation = explanation + f"{l2}\n"
    for line in list_line_neg:
        explanation += f"{line}\n"
    explanation = explanation + '\n'

    return explanation

def make_letter_pdf(base_latex, app_name, p1, n1, n2, n3):

    p1 = p1.replace('_', '\\_')
    n1 = n1.replace('_', '\\_')
    n2 = n2.replace('_', '\\_')
    n3 = n3.replace('_', '\\_')

    p1 = p1.replace('\\$', '$')
    n1 = n1.replace('\\$', '$')
    n2 = n2.replace('\\$', '$')
    n3 = n3.replace('\\$', '$')


    base_latex = base_latex.replace('RESERVEDAPPID1', str(app_name))
    base_latex = base_latex.replace('RESERVEDPOS1', str(p1))
    base_latex = base_latex.replace('RESERVEDNEG1', str(n1))
    base_latex = base_latex.replace('RESERVEDNEG2', str(n2))
    base_latex = base_latex.replace('RESERVEDNEG3', str(n3))

    pdf_object = build_pdf(base_latex)
    pdf_bytes = pdf_object.readb()

    return pdf_bytes

def process_apps(df_samp, model, df_ground, dict_r2f, dict_r2explain, dict_r2explain_positive, base_latex, tf = 'full'):
    """
    Perform full processing of the applications
    :param df_samp: dataframe of the applications
    :param model: saved catboost model
    :param df_ground: ground truth on which the model was trained
    :param base_latex: string.  input latex file for the customer letter.  Will be programatically modified to fit the
        given application
    :return:
        df_res = pandas dataframe with the decision results for each application
        explanations = string containing explanations of the decisions intended for the lender
        zip_bytes = bytes for a zip file containing letters to the declined applicants.
    """
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
    for appnum in df_samp.index:
        # print(f"{pd.to_datetime(time.time(), unit='s')}, Processing {i}")
        df_app = df_samp.loc[[appnum]]
        row, explanation, pdf_bytes = process_one(df_app=df_app,
                                                  model=model,
                                                  df_ground=df_ground,
                                                  dict_r2f=dict_r2f,
                                                  dict_r2explain=dict_r2explain,
                                                  dict_r2explain_positive=dict_r2explain_positive,
                                                  base_latex=base_latex,
                                                  appnum=appnum,
                                                  tf=tf)
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




def process_one(df_app, model, df_ground, base_latex, dict_r2f, dict_r2explain, dict_r2explain_positive, appnum = 0, tf = 'full'):
    # print(f"PROCESSING WITH TF = {tf}")
    df_app.reset_index(inplace=True)
    app_name = df_app.loc[0,'application_id']
    df_app = load_and_process(df_app, tf=tf)
    # print(df_app.shape)
    decision, prob = eval_app(model, df_app, tf=tf)
    shap_ser = get_shap_values(model, df_app, tf=tf)
    dict_pos, dict_neg = get_reasons_for_lender(df_ground, shap_ser, df_samp=df_app)
    explanation = make_explanation_string(decision, prob, dict_pos, dict_neg, app_name, appnum=appnum, tf=tf)
    if decision.upper() == 'DECLINE':
        # print(f"\n{app_name}")
        p1, n1, n2, n3 = get_reasons_for_applicant(shap_ser, dict_r2f, dict_r2explain, dict_r2explain_positive, how='mean')
        # print(f"{p1}\n{n1}\n{n2}\n{n3}")
    row = [app_name, decision, prob]
    for j in range(3):
        if j in dict_pos.keys():
            row.extend(dict_pos[j])
        else:
            row.extend(['None', np.nan, np.nan])
    for j in range(3):
        if j in dict_neg.keys():
            row.extend(dict_neg[j])
        else:
            row.extend(['None', np.nan, np.nan])

    if decision.upper() == 'DECLINE':
        pdf_bytes = make_letter_pdf(base_latex=base_latex,
                                    app_name=app_name,
                                    p1=p1,
                                    n1=n1,
                                    n2=n2,
                                    n3=n3)
    else:
        pdf_bytes = None


    return row, explanation, pdf_bytes

def score_field_list(shap_ser, field_list, how = 'sum'):
    """
    Calculated the aggregate score to the list of fields in field_list corresponding to how much they contributed to a pos or neg decision.
    :param shap_ser: Pandas Series of the Shapley values for each input variable of the model.  Index is the input variables.
    :param field_list: list of field names with respect to which we want the aggregate score.
    :return: sum or average of the shaplety values for variables in the field_list
    """
    field_list_present = [f for f in field_list if f in shap_ser.index]
    shap_ser_fields = shap_ser[field_list_present].copy()
    if len(shap_ser_fields) == 0:
        return 0
    if how == 'sum':
        return shap_ser_fields.sum()
    try:
        return shap_ser_fields.aggregate(how)
    except:
        return shap_ser_fields.mean()


def score_reasons(shap_ser, dict_r2f, how='mean'):
    ser_reason_score = pd.Series()
    for reason, field_list in dict_r2f.items():
        if reason in DUPLICATE_REASONS:
            continue
        ser_reason_score[reason] = score_field_list(shap_ser, field_list, how)
    # display(ser_reason_score)
    ser_reason_scorep = ser_reason_score[ser_reason_score >= 0].copy()
    ser_reason_scoren = ser_reason_score[ser_reason_score < 0].copy().abs()
    ser_reason_scorep.sort_values(ascending=False, inplace=True)
    ser_reason_scoren.sort_values(ascending=False, inplace=True)

    return ser_reason_scorep, ser_reason_scoren

def get_reasons_for_applicant(shap_ser, dict_r2f, dict_r2explain, dict_r2explain_positive, how='mean'):
    """
    TODO:  Deal with the unlikely case that a declined application has fewer than 3 negative reasons.
    Gets latex strings for the reasons to the applicant.
    :param shap_ser:
    :param dict_r2f:
    :param dict_r2explain:
    :param how:
    :return:
    """
    ser_reason_scorep, ser_reason_scoren = score_reasons(shap_ser, dict_r2f, how)
    n1 = ser_reason_scoren.index[0]
    n2 = ser_reason_scoren.index[1]
    n3 = ser_reason_scoren.index[2]
    if len(ser_reason_scorep) > 0:
        p1 = ser_reason_scorep.index[0]
    else:
        p1 = ser_reason_scoren.index[-1]

    # print(n1, n2, n3, p1)
    # return n1, n2, n3, p1

    n1 = dict_r2explain[n1]
    n2 = dict_r2explain[n2]
    n3 = dict_r2explain[n3]
    p1 = dict_r2explain_positive[p1]
    return p1, n1, n2, n3

def promote_hard_stops(dict_neg_orig, hard_stops = None):
    """
    Promotes the hard stop reason fields to the top of the order in dict_neg_orig as given by keys, after ordering.
    :param dict_neg_orig:
    :param hard_stops:
    :return:
    """
    if not hard_stops:
        hard_stops = HARD_STOPS.copy()
    fields = [[i, list(dict_neg_orig[i])] for i in range(len(dict_neg_orig))]
    # print(len(fields))
    # print(fields)
    hard_stops_present = [x for x in fields if x[1][0] in hard_stops]
    non_hard_present = [x for x in fields if x[1][0] not in hard_stops]
    # print(len(hard_stops_present))
    # print(hard_stops_present)
    hard_stops_present.sort(key = lambda x:x[0])
    if len(hard_stops_present) == 0 or len(hard_stops_present) == len(dict_neg_orig):
        return dict_neg_orig
    if len(hard_stops_present) == 1:
        h = hard_stops_present[0]
        old_pos = h[0]
        if old_pos == 1:
            fields[0][0] = 1
            fields[1][0] = 0
        if old_pos == 2:
            fields[0][0] = 1
            fields[1][0] = 2
            fields[2][0] = 0
    if len(hard_stops_present) == 2:
        non = non_hard_present[0]
        old_non = non[0]
        if old_non == 0:
            fields[0][0] = 2
            fields[1][0] = 0
            fields[2][0] = 1
        if old_non == 1:
            fields[1][0] = 2
            fields[2][0] = 1
    # print(f"NEW FIELDS {fields}")
    dict_neg_new = {}
    for x in fields:
        k = x[0]
        v = tuple(x[1])
        dict_neg_new[k] = v
    return dict_neg_new




