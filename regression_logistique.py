

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.metrics.pairwise import cosine_similarity


###################### Functions to implemente regression logistic  ############################

def logistic_regression_nlp(df, text_col="vlean_text_filtered", label_col="titulaire-soutien", min_docs=5, cv_folds=5, random_state=42):
    """
    Performs logistic regression on NLP text data with TF-IDF features using Stratified CV.

    Parameters:
        df : DataFrame containing text and labels
        min_docs : Minimum number of documents a word must appear in
        cv_folds : Number of folds for StratifiedKFold
        random_state : Random state for reproducibility

    Returns:
        best_model : trained Pipeline with the best hyperparameters and confusion matrice
    """

    X = df["clean_text_filtered"].astype(str)
    y = df[label_col].astype(str)

    # Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])


    # Hyperparameter grid
    param_grid = {
        'tfidf__max_features': [1000, 3000, 5000],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'lr__C': [0.1, 1.0, 10.0]
    }

    # GridSearch 
    #    # StratifiedKFold → conserve the proportion of each class in each fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X, y)

       # Best model
    best_model = grid.best_estimator_
    print("=== BEST PIPELINE ===")
    display(best_model)
    display(f"✅ Best hyperparameters: {grid.best_params_}, F1 (CV): {grid.best_score_:.4f}")

    # Cross-val predictions pour classification report et confusion matrix
    y_pred = cross_val_predict(best_model, X, y, cv=cv)
    print("\n--- CLASSIFICATION REPORT (CV) ---")
    print(classification_report(y, y_pred))

    fig, ax = plt.subplots(figsize=(20,10))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    plt.title("Confusion Matrix (CV)")
    plt.tight_layout()
    plt.show()

    return best_model

    



###################### Functions to interpret coefficients of logistic regression############################

def build_party_table_abs(coef_df, cat, top_n=10):
    """
    Build a table of the top |coef| features for a given party.

    Input: 
        - coef_df: DataFrame of coefficients
        - cat: party/class name
        - top_n: number of features to display

    Output: table with top_n features and their coefficients.
    """
    df_class = coef_df[coef_df['class'] == cat].copy()
    
    # Rank features by absolute coefficient magnitude
    df_class["abs_coef"] = df_class["coef"].abs()
    top_abs = df_class.sort_values("abs_coef", ascending=False).head(top_n)
    
    table = top_abs[["feature", "coef"]].reset_index(drop=True)
    table.columns = [f"{cat} - mot", f"{cat} - coef"]
    
    return table


def color_coef(val):
    """
    Assign color styling based on coefficient sign.

    Input: logistic regression coefficient.

    Output: CSS style (blue for positive, red for negative).
    """
    if val > 0:
        return "color: #1f77b4; font-weight: bold;"  # blue
    elif val < 0:
        return "color: #d62728; font-weight: bold;"  # red
    else:
        return ""


def display_three_parties_abs(coef_df, categories, n_cols=3, top_n=10):
    """
    Display styled tables of top |coef| features for parties,
    grouped by rows of n_cols parties.

    INPUT:
        - coef_df: DataFrame of coefficients (output of build_coef_df)
        - categories: list of party names.
        - n_cols: number of parties per row.
        - top_n: number of features per party

    OUTPUT:
        - None (displays styled tables)
    """
    for i in range(0, len(categories), n_cols):
        block = categories[i:i+n_cols]
        tables = [build_party_table_abs(coef_df, cat, top_n=top_n) for cat in block]
        combined = pd.concat(tables, axis=1)
        coef_cols = [col for col in combined.columns if "coef" in col]
        styled = combined.style.applymap(color_coef, subset=coef_cols).format(precision=3)
        display(styled)



################# Functions to transform the raw model components into accessible variables ###################

def extract_model_assets(model):
    """
    Extracts core components from the fitted pipeline.

    INPUT:
        - model : a trained pipeline containing
                * 'tfidf' (TfidfVectorizer)
                * 'lr' (LogisticRegression)

    OUTPUT:
        - assets: Dictionary containing:
            * 'feature_names' (np.array): Vocabulary used by the TF-IDF vectorizer.
            * 'coefs' (np.array): Logistic regression coefficient matrix 
                                  (shape: n_classes × n_features).
            * 'classes' (np.array): Ordered class labels used by the classifier.
            * 'vectorizer' (TfidfVectorizer): Fitted TF-IDF vectorizer object.
            * 'classifier' (LogisticRegression): Fitted logistic regression model.
    """
    # Access steps from the pipeline
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["lr"]
    
    # Extract metadata
    assets = {
        "feature_names": vectorizer.get_feature_names_out(),
        "coefs": classifier.coef_,
        "classes": classifier.classes_,
        "vectorizer": vectorizer,
        "classifier": classifier
    }
    return assets


def build_coef_df(assets):
    """
    Constructs a tidy DataFrame of coefficients for each feature and class.
    
    INPUT:
        - assets: Dictionary containing 'coefs', 'feature_names', and 'classes'.
    
    OUTPUT:
        - coef_df: Tidy DataFrame with columns ['class', 'feature', 'coef'].
    """
    # Create the DataFrame directly from the coefficients matrix
    # rows = classes, columns = features
    df_c = pd.DataFrame(
        assets["coefs"], 
        index=assets["classes"], 
        columns=assets["feature_names"]
    )
    
    # "Melt" the dataframe to get it into a tidy format (long format) easier to manipulate
    df_c = df_c.reset_index().melt(
        id_vars='index', 
        var_name='feature', 
        value_name='coef'
    )
    df_c.columns = ['class', 'feature', 'coef']
    
    return df_c


################# Funtion to run the heavy computations once ####################"###

# TF-IDF Transformation
def transform_texts(df, vectorizer, text_col):
    """
    Transform raw texts into TF-IDF vectors.
    
    INPUT:
        - df (pd.DataFrame): DataFrame containing the texts
        - vectorizer (TfidfVectorizer): fitted TF-IDF vectorizer
        - text_col (str): column name containing the text
    
    OUTPUT:
        - X (np.array or sparse matrix): TF-IDF feature matrix
    """
    return vectorizer.transform(df[text_col])


# Model Predictions
def compute_predictions(df, classifier, X, label_col):
    """
    Compute predicted labels, misclassification flags, and probabilities.
    
    INPUT:
        - df (pd.DataFrame)
        - classifier (LogisticRegression)
        - X: TF-IDF matrix
        - label_col: name of true label column
    
    OUTPUT:
        - df_pred (pd.DataFrame): df enriched with:
            * predicted_label
            * is_misclassified
            * probability columns for each class
    """
    df_pred = df.copy().reset_index(drop=True)
    probas = classifier.predict_proba(X)
    df_pred['predicted_label'] = classifier.predict(X)
    df_pred['is_misclassified'] = df_pred[label_col] != df_pred['predicted_label']
    
    # Add probability columns
    proba_df = pd.DataFrame(probas, columns=classifier.classes_)
    df_pred = pd.concat([df_pred, proba_df], axis=1)
    
    return df_pred, probas


# t-SNE Embedding

def compute_tsne(df, probas, random_state=42, perplexity=30):
    """
    Compute 2D t-SNE embedding from class probabilities.
    
    INPUT:
        - df (pd.DataFrame)
        - probas (np.array)
    
    OUTPUT:
        - df_tsne (pd.DataFrame): df enriched with columns 'tsne_1', 'tsne_2'
    """
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    tsne_results = tsne.fit_transform(probas)
    df['tsne_1'] = tsne_results[:, 0]
    df['tsne_2'] = tsne_results[:, 1]
    return df

# Full pipeline
def build_master_df(df, assets, text_col, label_col):
    """
    Full pipeline to compute all model metrics and distances.
    
    INPUT:
        - df (pd.DataFrame)
        - assets (dict): output of extract_model_assets()
        - text_col, label_col (str)
    
    OUTPUT:
        - df_master (pd.DataFrame): enriched DataFrame with predictions, probabilities, t-SNE, and distances
        - X (np.array): TF-IDF matrix
    """
    X = transform_texts(df, assets["vectorizer"], text_col).toarray()
    df_master, probas = compute_predictions(df, assets["classifier"], X, label_col)
    df_master = compute_tsne(df_master, probas)
    df_master = compute_centroid_distances(df_master, X, assets["classes"], label_col)
    return df_master, X


# Centroid Distances
def compute_centroid_distances(df, X, classes, label_col):
    """
    Compute cosine distance from each document to its class centroid.
    
    INPUT:
        - df (pd.DataFrame)
        - X (np.array): TF-IDF matrix
        - classes (list/array): class labels
        - label_col (str)
    
    OUTPUT:
        - df_dist (pd.DataFrame): df enriched with 'distance_to_centroid'
    """
    
    df['distance_to_centroid'] = np.nan
    for cls in classes:
        mask = (df[label_col] == cls).values 
        if not any(mask):
            continue
        vectors = X[mask]
        centroid = vectors.mean(axis=0).reshape(1, -1)
        df.loc[mask, 'distance_to_centroid'] = cdist(vectors, centroid, metric='cosine').flatten()
    return df


# Intra-party distance

def compute_intra_party_distances(df, X, label_col):
    """
    Computes the average intra-party distance for each party based on cosine similarity.

    INPUTS:
        - df (pd.DataFrame): DataFrame containing the party/category labels.
        - X (np.array or sparse matrix): Document-feature matrix (e.g., TF-IDF vectors).
        - label_col (str): Name of the column in df containing party/category labels.

    OUTPUT:
        - intra_distances (dict): Dictionary where keys are party names and values 
          are the average cosine distance between documents within that party.
          Example: {'Party A': 0.12, 'Party B': 0.34, ...}
    """
    # Ensure X is an array if it's a np.matrix
    if isinstance(X, np.matrix):
        X = np.asarray(X)
    
    # Convert sparse matrix to CSR if possible
    X = X.tocsr() if hasattr(X, "tocsr") else X
    parties = df[label_col].unique()
    intra_distances = {}

    for p in parties:
        mask = (df[label_col] == p).values
        vectors = X[mask]

        if vectors.shape[0] < 2:
            # Not enough documents to compute distance
            intra_distances[p] = 0.0
            continue

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(vectors)

        # Take the upper triangle only to ignore self-similarity
        upper_indices = np.triu_indices(sim_matrix.shape[0], k=1)
        mean_sim = sim_matrix[upper_indices].mean()

        # Cosine distance = 1 - similarity
        intra_distances[p] = 1 - mean_sim

    return intra_distances
