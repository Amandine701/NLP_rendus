import textstat

from create_dataframe import prepare_legislatives_dataset_lisibilite
from functions_to_clean_data import clean_for_readability

# Configuration 
textstat.set_lang("fr")

def process_year(csv_path, txt_path):
    """ Loads data and computes readability scores for a specific year. """
    df = prepare_legislatives_dataset_lisibilite(csv_path, txt_path)
    # Apply light cleaning and compute metrics
    df['text_clean_light'] = df['text_content'].apply(clean_for_readability)
    df['flesch_score'] = df['text_clean_light'].apply(textstat.flesch_reading_ease)
    df['flesch_kincaid_grade'] = df['text_clean_light'].apply(textstat.flesch_kincaid_grade)
    df['gunning_fog'] = df['text_clean_light'].apply(textstat.gunning_fog)
    return df