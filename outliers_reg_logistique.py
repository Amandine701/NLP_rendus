import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.neighbors import KNeighborsClassifier
from regression_logistique import compute_centroid_distances
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances

######################### Function to detect "centroid outliers" (cosine distance) ###########################

def detect_centroid_outliers(df, model, label_col='titulaire-soutien', text_col='clean_text_filtered', iqr_factor=1.5):
    """
    Computes cosine distance to party centroids and identifies outliers per party using the IQR rule.

    INPUTS:
        - df (pd.DataFrame): DataFrame containing the text and party/category labels.
        - model (sklearn.pipeline.Pipeline): Trained pipeline including a 'tfidf' step.
        - label_col (str): Name of the column containing the party/category labels.
        - text_col (str): Name of the column containing the preprocessed text.
        - iqr_factor (float, default=1.5): Multiplier for the IQR to define the upper bound for outliers.

    OUTPUTS:
        - df (pd.DataFrame): Original DataFrame updated with:
            * 'distance_au_centre' (float): Cosine distance of each document to its party centroid.
            * 'is_outlier_centroid' (bool): Flag indicating whether the document is an outlier.
        - df_outliers (pd.DataFrame): Filtered DataFrame containing only the documents flagged as outliers.
    """
    
    # Extract TF-IDF matrix; convert to CSR for efficient row masking
    tfidf = model.named_steps['tfidf']
    X = tfidf.transform(df[text_col]).tocsr()
    
    classes = df[label_col].unique()
    df['distance_au_centre'] = np.nan
    
    # Compute cosine distance to centroid for each class
    for cls in classes:
        mask = (df[label_col] == cls).values 
        if not any(mask): 
            continue
            
        vectors = X[mask]
        
        # Compute centroid as mean vector (converted to standard array)
        centroid = np.asarray(vectors.mean(axis=0))
        
        # Compute cosine distances from each document to centroid
        dists = cosine_distances(vectors, centroid).flatten()
        df.loc[mask, 'distance_au_centre'] = dists

    # Identify outliers using IQR rule
    df['is_outlier_centroid'] = False
    for parti in classes:
        mask_parti = (df[label_col] == parti) & (df['distance_au_centre'].notna())
        subset_dist = df.loc[mask_parti, 'distance_au_centre']
        
        if len(subset_dist) < 4:  # Skip small groups
            continue
            
        Q1 = subset_dist.quantile(0.25)
        Q3 = subset_dist.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (iqr_factor * IQR)
        
        # Flag documents exceeding the IQR threshold as outliers
        df.loc[mask_parti & (df['distance_au_centre'] > upper_bound), 'is_outlier_centroid'] = True

    # Return the full DataFrame and a filtered DataFrame with only outliers
    return df, df[df['is_outlier_centroid']].copy()

######################### Function to detect "knn outliers" (cosine distance) ##########################

def detect_knn_outliers(df, k=10, label_col='titulaire-soutien', coord_cols=['tsne_1', 'tsne_2']):
    """
    Identifies outliers based on local neighborhood in t-SNE space using K-Nearest Neighbors.

    Inputs:
        df         : Dataframe containing t-SNE coordinates and labels.
        k          : Number of neighbors to consider.
        label_col  : Name of the column containing the true party labels.
        coord_cols : List of columns representing the spatial coordinates.

    Outputs:
        df         : Original dataframe updated with 'parti_geographique' and 'is_outlier_knn' (bool).
        df_outliers: Filtered dataframe containing only neighborhood outliers.
    """
    coords = df[coord_cols].values
    labels = df[label_col].values

    # Initialize and fit K-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(coords, labels)

    # Predict the majority party in the neighborhood
    df['parti_geographique'] = knn.predict(coords)

    # An outlier is a document whose label differs from its geographical neighbors
    df['is_outlier_knn'] = df[label_col] != df['parti_geographique']

    # Filtered dataframe for specific analysis
    df_outliers = df[df['is_outlier_knn'] == True].copy()

    return df, df_outliers


def detect_proba_outliers(df, model, threshold=0.30, label_col='titulaire-soutien', text_col='clean_text_filtered'):
    """
    Identifies outliers where the model's confidence in the true label is below a threshold.

    Inputs:
        df        : Dataframe containing the text and labels.
        model     : Trained Scikit-Learn pipeline (must support predict_proba).
        threshold : Probability threshold below which a document is an outlier.
        label_col : Name of the column with true labels.
        text_col  : Name of the column with preprocessed text.

    Outputs:
        df         : Original dataframe updated with 'proba_parti_reel' and 'is_outlier_proba' (bool).
        df_outliers: Filtered dataframe containing only low-confidence outliers.
    """
    #  Get probabilities for all classes
    probas = model.predict_proba(df[text_col])
    df_probas = pd.DataFrame(probas, columns=model.classes_, index=df.index)

    #  Extract the probability assigned to the TRUE party
    # Vectorized approach for performance
    df['proba_parti_reel'] = [df_probas.iloc[i][df.iloc[i][label_col]] for i in range(len(df))]

    #  Identify outliers (low confidence in the real label)
    df['is_outlier_proba'] = df['proba_parti_reel'] < threshold

    # Filtered dataframe
    df_outliers = df[df['is_outlier_proba'] == True].copy()

    return df, df_outliers


########################### Proportion of misclassification among outliers per party########################

def display_centroid_outliers_report(df_full, df_outliers, model, text_col='clean_text_filtered', label_col="titulaire-soutien", method_name="Centroid"):
    """
    Computes the proportion of misclassified documents within a specific outlier subset.

    Inputs:
        df_full     : Full DataFrame 
        df_outliers : DataFrame with only outliers
        model       : Pipeline trained for predictions
        text_col    : Text column
        label_col   : Labels's column
        method_name : Method name (centroid, knn, probas)

    Outputs:
        stats_outliers (pd.DataFrame): Summary table of outliers and their proportion of misclassification
    """
    
    # Generating predictions for outliers
    df_outliers = df_outliers.copy()
    df_outliers['parti_predit'] = model.predict(df_outliers[text_col])

    # Comparing predictions and true labels
    df_outliers['est_mal_classe'] = df_outliers[label_col] != df_outliers['parti_predit']

    # Proportion of outliers' misclassification per category
    stats_outliers = df_outliers.groupby(label_col).agg(
        total_outliers=(label_col, 'count'),
        nb_mal_classes=('est_mal_classe', 'sum')
    )

    stats_outliers['proportion_misclassification_%'] = (
        stats_outliers['nb_mal_classes'] / stats_outliers['total_outliers'] * 100
    ).round(2)

    print(f"=== Proportion of {method_name} outliers' misclassification per category ===")
    display(stats_outliers.sort_values(by='total_outliers', ascending=False))
    
    return stats_outliers


def rapport_outliers_style(df_outliers, X_matrix, model, feature_names, classes, top_n=5, max_rows=20):
    """
    Analyze outliers by generating predictions and identifying 
    anchor words (supporting the true party) and defector words 
    (pulling toward the challenger party).

       INPUTS:
    - df_outliers (df) : DataFrame containing outlier observations ( 'id', 'titulaire-soutien','clean_text_filtered')
    - X_matrix :TF-IDF matrix corresponding to df_outliers.
        Each row must align with df_outliers order.
    - model : sklearn Pipeline containing:
            - 'tfidf': TfidfVectorizer
            - 'lr': linear classifier with .coef_ and .predict_proba()
    - feature_names (list of str) :List of vocabulary terms corresponding to columns of X_matrix.
    - classes (list of str):  List of class labels in the same order as model coefficients.
    - top_n (int) Number of top contributing words to extract for anchors and defectors.

    OUTPUT:
    - pd.io.formats.style.Styler df containing, for each outlier:
            - ID
            - True party
            - Predicted party
            - Challenger (2nd most probable class with probability)
            - Anchor words (supporting the true class)
            - Defector words (supporting the challenger class)
            - Status (correct or misclassified)
        Misclassified rows are displayed in bold red.
        Correct rows are displayed in blue.
    """
    all_reports = []

    df_outliers = df_outliers.iloc[:max_rows]
    
    # Extract model components
    tfidf = model.named_steps['tfidf']
    lr = model.named_steps['lr']
    coeffs = lr.coef_
    
     # Compute predicted probabilities and predictions for outliers
    probas = model.predict_proba(df_outliers['clean_text_filtered'])
    preds = model.predict(df_outliers['clean_text_filtered'])
    df_probas = pd.DataFrame(probas, columns=classes, index=df_outliers.index)

    for i, (idx, row) in enumerate(df_outliers.iterrows()):
        vrai_parti = row['titulaire-soutien']
        parti_predit = preds[i]
        idx_vrai = list(classes).index(vrai_parti)
        proba_vrai = df_probas.loc[idx, vrai_parti]  # probability of the true class
        
        # Identify the Challenger (2nd most probable party)
        row_probas = df_probas.iloc[i].sort_values(ascending=False)
        # Challenger is 2nd if model is correct, 1st if model is wrong
        challenger_parti = row_probas.index[1] if row_probas.index[0] == vrai_parti else row_probas.index[0]
        idx_challenger = list(classes).index(challenger_parti)

        # Convert sparse row to dense array
        doc_vector = tfidf.transform([row['clean_text_filtered']]).toarray().flatten()
        
        # Compute contributions
        contrib_vrai = doc_vector * coeffs[idx_vrai]
        ancres_idx = np.argsort(contrib_vrai)[::-1][:top_n]
        ancres = [feature_names[j] for j in ancres_idx if contrib_vrai[j] > 0]

        contrib_challenger = doc_vector * coeffs[idx_challenger]
        trans_idx = np.argsort(contrib_challenger)[::-1][:top_n]
        transfuges = [feature_names[j] for j in trans_idx if contrib_challenger[j] > 0]

        all_reports.append({
            "ID": row['id'],
            "Vrai Parti": f"{vrai_parti} ({proba_vrai:.1%})",
            "Prédit": parti_predit,
            "Challenger (2nd)": f"{challenger_parti} ({row_probas[challenger_parti]:.1%})",
            "Mots Ancres (Confirmant)": ", ".join(ancres),
            "Mots Transfuges (Vers Challenger)": ", ".join(transfuges),
            "Statut": "❌ Erreur" if vrai_parti != parti_predit else "✅ Atopique"
        })

    df_report = pd.DataFrame(all_reports)
    
    # Final styling
    return df_report.style.set_properties(**{'text-align': 'left'})\
        .apply(lambda x: ['color: red; font-weight: bold' if v == "❌ Erreur" else 'color: green' 
                          for v in x], subset=['Statut'], axis=0)

