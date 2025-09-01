import pandas as pd
import numpy as np

COLUMNS = ['intake_intake_channel', 'intake_new_customer', 'intake_lender_segment',
       'intake_region', 'intake_sector', 'intake_business_age_years',
       'intake_annual_revenue_gbp', 'intake_requested_loan_gbp',
       'intake_requested_term_months', 'intake_loan_purpose',
       'intake_personal_guarantee', 'intake_collateral_type',
       'intake_bureau_score', 'intake_ccj_count', 'intake_default_history',
       'intake_kyc_pass', 'intake_aml_risk', 'intake_pep_hit',
       'intake_sanctions_hit', 'intake_doc_completeness', 'enrich_dscr',
       'enrich_leverage_ratio', 'enrich_cashflow_volatility', 'enrich_ltv',
       'enrich_risk_score', 'enrich_triage_priority', 'label']
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


def load_and_process(filepath):
    columns = COLUMNS
    cat_features = CAT_FEATURES
    nums = NUMS
    dropcols = DROPCOLS

    df = pd.read_csv(filepath)
    for c in dropcols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    try:
        df = df[columns]
    except:
        print('Missing columns')

    for c in df.columns:
        if not c in nums:
            df[c] = df[c].fillna(value='None')

    return df





