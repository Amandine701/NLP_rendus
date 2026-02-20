# This file creates a unified and cleaned dataframe based on the results of the descriptive statistical analysis.

def prepare_legislatives_dataset(csv_path, txt_folder_path, threshold=30):
    import pandas as pd
    import os
    """
    This function creates a unified, cleaned and exploitable dataframe

        INPUTS:
        - csv_path (str) :Path to the CSV file containing metadata.
        - txt_folder_path (str): Path to the folder containing text files. 
        - threshold : Minimum number of documents per 'titulaire-soutien' to keep the class.

        OUTPUTS:df_filtered (df) : Cleaned dataframe 
    """
    
    ## DATA LOADING & INITIAL CLEANING 
    # Load original dataset
    csv = pd.read_csv(csv_path, encoding="utf-8")

    # Select relevant columns and drop rows with invalid 'titulaire-soutien' labels
    df_reduced = csv[["id", "contexte-tour", "titulaire-soutien", "date"]].copy()
    df_valid = df_reduced[
        (~(df_reduced['titulaire-soutien'].isna() | 
        (df_reduced['titulaire-soutien'] == '') | 
        (df_reduced['titulaire-soutien'] == 'non mentionné')))
        & (df_reduced['contexte-tour'] == 1)
        ].copy()


    ##  TEXT CONTENT INTEGRATION 
    path_to_files = txt_folder_path
    txt_files = [f for f in os.listdir(path_to_files) if f.endswith(".txt")]

    # Extract ID and text content from files
    file_data = [{"id": f.replace(".txt", ""), "text_content": open(os.path.join(path_to_files, f), 'r', encoding='utf-8').read()} 
                 for f in txt_files]

    # Merge metadata with text content
    df_txt = pd.DataFrame(file_data)
    df_valid = pd.merge(df_valid, df_txt, on="id", how="left")

    ##  FREQUENCY FILTERING (N >= 30) 
    # Define threshold for statistical significance
    SEUIL_MIN = threshold
    df_filtered = df_valid.groupby("titulaire-soutien").filter(lambda x: len(x) >= SEUIL_MIN).copy()
    df_filtered = df_filtered.reset_index(drop=True)

    # Df filtered
    return df_filtered

def prepare_legislatives_dataset_without_bi_categories(csv_path, txt_folder_path, threshold=30):
    import pandas as pd
    import os
    """
    This function creates a unified, cleaned and exploitable dataframe (without bi-categories)
     INPUTS:
        - csv_path (str) :Path to the CSV file containing metadata.
        - txt_folder_path (str): Path to the folder containing text files. 
        - threshold : Minimum number of documents per 'titulaire-soutien' to keep the class.

        OUTPUTS:df_filtered (df) : Cleaned dataframe without bicategories
    """
    
    ## DATA LOADING & INITIAL CLEANING 
    # Load original dataset
    csv = pd.read_csv(csv_path, encoding="utf-8")

    # Select relevant columns and drop rows with invalid 'titulaire-soutien' labels
    df_reduced = csv[["id", "contexte-tour", "titulaire-soutien", "date"]].copy()
    df_valid = df_reduced[
        (~(df_reduced['titulaire-soutien'].isna() | 
        (df_reduced['titulaire-soutien'] == '') | 
        (df_reduced['titulaire-soutien'] == 'non mentionné')))
        & (df_reduced['contexte-tour'] == 1)
        ].copy()

    ##  TEXT CONTENT INTEGRATION 
    path_to_files = txt_folder_path
    txt_files = [f for f in os.listdir(path_to_files) if f.endswith(".txt")]

    # Extract ID and text content from files
    file_data = [{"id": f.replace(".txt", ""), "text_content": open(os.path.join(path_to_files, f), 'r', encoding='utf-8').read()} 
                 for f in txt_files]

    # Merge metadata with text content
    df_txt = pd.DataFrame(file_data)
    df_valid = pd.merge(df_valid, df_txt, on="id", how="left")

    ##  FREQUENCY FILTERING (N >= 30) 
    # Define threshold for statistical significance
    SEUIL_MIN = threshold
    df_filtered = df_valid.groupby("titulaire-soutien").filter(lambda x: len(x) >= SEUIL_MIN).copy()
    df_filtered = df_filtered.reset_index(drop=True)

      ## REMOVE BI-SUPPORT CATEGORIES
    categories_a_supprimer = [
        'Parti socialiste;Mouvement des radicaux de gauche', 
        'Rassemblement pour la République;Union pour la démocratie française',  
        'Union pour la démocratie française;Rassemblement pour la République'
    ]
    df_filtered = df_filtered[~df_filtered['titulaire-soutien'].isin(categories_a_supprimer)].copy()
    df_filtered = df_filtered.reset_index(drop=True)

    # Update working dataframe for downstream tasks (splits, training)
    return df_filtered

def prepare_legislatives_dataset_lisibilite(csv_path, txt_folder_path, threshold=30):
    import pandas as pd
    import os
    """
    This function is the same as prepare_legislatives_dataset except we don't 
    remove manifestos without identified support and parties with less than 30 manifestos
    """
    
    ## DATA LOADING & INITIAL CLEANING 
    # Load original dataset
    csv = pd.read_csv(csv_path, encoding="utf-8")

    # Select relevant columns and drop rows with invalid 'titulaire-soutien' labels
    df_reduced = csv[["id", "contexte-tour", "titulaire-soutien", "date"]].copy()
    df_valid = df_reduced[(df_reduced['contexte-tour'] == 1)
        ].copy()


    ##  TEXT CONTENT INTEGRATION 
    path_to_files = txt_folder_path
    txt_files = [f for f in os.listdir(path_to_files) if f.endswith(".txt")]

    # Extract ID and text content from files
    file_data = [{"id": f.replace(".txt", ""), "text_content": open(os.path.join(path_to_files, f), 'r', encoding='utf-8').read()} 
                 for f in txt_files]

    # Merge metadata with text content
    df_txt = pd.DataFrame(file_data)
    df_valid = pd.merge(df_valid, df_txt, on="id", how="left")

    # Df filtered
    return df_filtered