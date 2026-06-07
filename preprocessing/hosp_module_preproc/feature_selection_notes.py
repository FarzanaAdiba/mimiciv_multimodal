from '..' import notes_preproc
from notes_preproc import *




def feature_notes(diag_flag, med_hist, f_med_hist, hist_drug, hist_mi, df, pred):
    selected_labels = []

    if diag_flag:
        selected_labels.append("DIAGNOSIS")
    if med_hist:
        selected_labels.append("MEDICATION") 
    if f_med_hist:
        selected_labels.append("FAMILY MEDICAL HISTORY")
    if hist_drug:
        selected_labels.append("DRUG-USE")
    if hist_mi:
        selected_labels.append("MENTAL ILLNESS") 

    if pred == 'Negation Detection' or pred == 'Severity Detection':
        df = df[df['label'].isin(selected_labels)]

    return df