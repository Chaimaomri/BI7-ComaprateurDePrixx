import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import os
import csv

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def detect_delimiter(file_path, encodings=['utf-8', 'ISO-8859-1', 'latin1']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                first_line = f.readline().strip()
                logging.debug(f"First line of {file_path}: {first_line}")
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                tab_count = first_line.count('\t')
                if semicolon_count > comma_count and semicolon_count > tab_count:
                    return ';'
                elif tab_count > comma_count and tab_count > semicolon_count:
                    return '\t'
                else:
                    return ','
        except UnicodeDecodeError as e:
            logging.error(f"UnicodeDecodeError while detecting delimiter with encoding {encoding}: {str(e)}")
            continue
    return ','  # Fallback to comma if detection fails

def get_data_from_csv(file_achats, file_produits, file_clients):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    achats, produits, clients = None, None, None
    
    # Detect delimiters for each file
    delimiter_achats = detect_delimiter(file_achats)
    delimiter_produits = detect_delimiter(file_produits)
    delimiter_clients = detect_delimiter(file_clients)
    
    logging.debug(f"Detected delimiters: achats={delimiter_achats}, produits={delimiter_produits}, clients={delimiter_clients}")
    
    for encoding in encodings:
        try:
            achats = pd.read_csv(file_achats, encoding=encoding, delimiter=delimiter_achats, decimal=',')
            produits = pd.read_csv(file_produits, encoding=encoding, delimiter=delimiter_produits, on_bad_lines='skip', decimal=',')
            clients = pd.read_csv(file_clients, encoding=encoding, delimiter=delimiter_clients, decimal=',')
            logging.debug(f"Successfully read achats with columns: {achats.columns.tolist()}")
            logging.debug(f"Successfully read produits with columns: {produits.columns.tolist()}")
            logging.debug(f"Successfully read clients with columns: {clients.columns.tolist()}")
            break
        except UnicodeDecodeError as e:
            logging.error(f"UnicodeDecodeError with encoding {encoding}: {str(e)}")
            continue
        except pd.errors.ParserError as e:
            logging.error(f"ParserError with encoding {encoding}: {str(e)}")
            continue
    
    if achats is None or produits is None or clients is None:
        raise ValueError("Unable to read CSV files. Try saving them with UTF-8 encoding and a standard delimiter (e.g., comma).")
    
    return achats, produits, clients

def build_models(file_achats, file_produits, file_clients):
    # Load data
    achats, produits, clients = get_data_from_csv(file_achats, file_produits, file_clients)
    
    # Validate required columns
    required_achats_cols = ['Client_FK', 'Produit_FK', 'Quantité_Achetée']
    required_produits_cols = ['ProduitID', 'Produit']
    required_clients_cols = ['Client_PK', 'Nom_Client']
    
    # Log raw and processed columns
    logging.debug(f"Raw achats columns: {achats.columns.tolist()}")
    achats_cols = [col.strip().lower() for col in achats.columns]  # Strip spaces from column names
    logging.debug(f"Processed lowercase achats columns: {achats_cols}")
    if not all(col.lower() in achats_cols for col in required_achats_cols):
        missing = [col for col in required_achats_cols if col.lower() not in achats_cols]
        logging.error(f"Missing columns in achats: {missing}")
        raise ValueError(f"fichier_achats missing required columns: {missing}")
    if not all(col.lower() in [c.lower() for c in produits.columns] for col in required_produits_cols):
        missing = [col for col in required_produits_cols if col.lower() not in [c.lower() for c in produits.columns]]
        raise ValueError(f"fichier_produits missing required columns: {missing}")
    if not all(col.lower() in [c.lower() for c in clients.columns] for col in required_clients_cols):
        missing = [col for col in required_clients_cols if col.lower() not in [c.lower() for c in clients.columns]]
        raise ValueError(f"fichier_clients missing required columns: {missing}")
    
    # Convert ProduitID to int64 if possible
    produits['ProduitID'] = pd.to_numeric(produits['ProduitID'], errors='coerce').astype('Int64')  # Use Int64 to handle NaN
    
    # Filter out invalid entries
    achats = achats[achats['Produit_FK'] != 0]
    logging.debug(f"Filtered achats shape: {achats.shape}, columns: {achats.columns.tolist()}")
    
    # Merge data
    df = achats.merge(produits, left_on="Produit_FK", right_on="ProduitID", how='left')
    df = df.merge(clients, left_on="Client_FK", right_on="Client_PK", how='left')
    
    # Create pivot table
    pivot = df.pivot_table(index='Nom_Client', columns='Produit', values='Quantité_Achetée', aggfunc='sum').fillna(0)
    if pivot.empty:
        raise ValueError("No valid data available for pivot table creation.")
    
    # Train KNN model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(pivot.values)
    
    return pivot, model_knn

def get_clients_similaires(client_nom, pivot, model_knn, n=4):
    try:
        index = pivot.index.get_loc(client_nom)
        _, indices = model_knn.kneighbors([pivot.iloc[index].values], n_neighbors=n+1)
        similar_clients = [pivot.index[i] for i in indices[0] if pivot.index[i] != client_nom][:n]
        return similar_clients
    except KeyError:
        raise ValueError(f"Client '{client_nom}' not found in data")

def score_hybride(client_nom, pivot, similaires, alpha=0.6, top_n=10):
    try:
        # Products bought by the client
        produits_achetes = pivot.loc[client_nom][pivot.loc[client_nom] > 0].index.tolist()
        
        # Products bought by similar clients
        produits_sim = pivot.loc[similaires]
        popularite = produits_sim.sum().drop(labels=produits_achetes, errors='ignore')
        
        # Generate recommendations
        recommandations = []
        for produit in popularite.index:
            score = popularite[produit]  # Simple popularity score
            recommandations.append((produit, score))
        
        return sorted(recommandations, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        raise ValueError(f"Error generating recommendations for {client_nom}: {str(e)}")