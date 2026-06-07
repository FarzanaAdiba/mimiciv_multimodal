#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'MIMIC-IV-Data-Pipeline-main', 'mimiciv', 'notes'))
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
sys.path.append(os.path.abspath("../utils"))
import icd_search
from icd_search import *
import re
import mimiciv_text_sectionizer
from mimiciv_text_sectionizer import *

import combination_util
from combination_util import *

sys.path.append("model/embedding_model")
import mimic_notes_embedding_BioClinicalBERT
from mimic_notes_embedding_BioClinicalBERT import *

import mimic_notes_embedding_BioSimCSE_BioLinkBERT
from mimic_notes_embedding_BioSimCSE_BioLinkBERT import *

import mimic_notes_embedding_gatortron
from mimic_notes_embedding_gatortron import *

import mimic_notes_embedding_ModernBERT
from mimic_notes_embedding_ModernBERT import *


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _choose_model(type_embedding, sec):
    match type_embedding:
        case "BioClinicalBERT":
            mimic_notes_embedding_BioClinicalBERT(sec)
        case "BioSimCSE_BioLinkBERT":
            mimic_notes_embedding_BioSimCSE_BioLinkBERT(sec)
        case "gatortron":
            mimic_notes_embedding_gatortron(sec)
        case "ModernBERT":
            mimic_notes_embedding_ModernBERT(sec)
        



def get_csv(sec):
    '''
    Open .csv file from what section the user chose.

    Arg:
        sec (str) - section the user chose
    Returns:
        pandas Dataframe - the opened .csv file
    '''

        
    try:
        return pd.read_csv(base_dir + '/mimiciv/notes/' + sec.lower() + '.csv.gz', compression='gzip', header=0, index_col=None)
    except Exception as e:
        raise FileNotFoundError(f'CSV file not found: {base_dir}/mimiciv/notes/{sec.lower()}.csv.gz\n\n{e}')

def section_splitting(df, sec):
    '''
    Calls mimic_text_sectionizer.py

    Arg: 
        df (DataFrame) - opened .csv note file
        sec 
    Returns:
        pandas DataFrame - df['text'] sectionized based off headers
    '''
    #using text_sectionizer created by Farzana
    print('Section splitting...')
    return mimiciv_text_sectionizer.extract_data_sectionizer(df, sec), 'SUMMARY PLACEHOLDER'
    

def extract_data(sec, pred, embedding, type_embedding, mimiciv_version, subject_ids=None):
    
    print('===========MIMIC-IV Notes============')
    print(f'Filtering: {subject_ids}')

    df = get_csv(sec)  # removed cohort argumen

    # Filter by subject_ids if provided

    if subject_ids is not None:
        df['subject_id'] = df['subject_id'].astype(int)
        subject_ids = [int(x) for x in subject_ids]
        df = df[df['subject_id'].isin(subject_ids)]
        print(f"Filtered notes to {len(df)} subjects based on provided subject_ids.")

    
    # If empty after filtering, return empty df with expected columns
    if df.empty:
        print("No matching subjects found. Returning empty DataFrame.")
        columns = ["subject_id","hadm_id","note_seq","charttime","storetime","note_id","section_id","section_name","text"]
        return pd.DataFrame(columns=columns)

    summary = f'EXTRACTING FOR: {sec.upper()}, {pred.upper()}'

    f_df, sub_summary = prediction(pred, df, sec)
    summary += '\n' + (sub_summary or '')

    with open(base_dir + '/data/cohort/notes_cohort_summary.txt', "w") as f:
        f.write(summary)
    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    print(summary)

    if embedding:
        print('Embedding notes...')
        _choose_model(type_embedding, sec.lower())

    f_df.to_csv(base_dir + "/data/cohort/mimiciv_notes_cohort.csv.gz", index=False, compression='gzip')
    print("[ COHORT SUCCESSFULLY SAVED ]")

    return f_df


    

def prediction(pred, df, sec):
    match pred:
        case 'Section':
            return section_splitting(df, sec)
        case 'Raw Text':
            return df, None
    


            





    