from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
from models.AlgoNonSup import prepare_data, run_clustering
from models.SerieTemporelle import run_forecasting
from models.recommendation import build_models, get_clients_similaires, score_hybride
from werkzeug.utils import secure_filename
import io
import base64
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import secrets
import matplotlib.pyplot as plt
import numpy as np
import logging
import traceback
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
from io import BytesIO

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)  # Autorise les requêtes CORS pour Angular

# Configuration du logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Configuration du dossier d'upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Définition des fonctions utilitaires
def generate_comment(note):
    positive_comments = [
        "Super séjour, tout était parfait.",
        "Très bon service, chambre propre.",
        "Le personnel était très accueillant.",
        "J'ai adoré cet hôtel, tout s'est bien passé.",
        "Excellent rapport qualité/prix.",
        "Magnifique vue depuis la chambre.",
        "Service rapide et efficace.",
        "Très bon accueil à la réception.",
        "Je recommande fortement cet endroit.",
        "Parfait pour un séjour en famille."
    ]
    neutral_comments = [
        "Séjour correct, sans plus.",
        "C'était moyen, rien de spécial.",
        "Ni bon ni mauvais.",
        "Service acceptable, un peu lent.",
        "Une expérience ordinaire.",
        "Hôtel correct mais un peu bruyant.",
        "Pas très impressionné, mais pas déçu non plus.",
        "Peu de différences par rapport aux autres hôtels.",
        "Manque de personnalité mais fonctionnel.",
        "Chambre simple, basique mais propre."
    ]
    negative_comments = [
        "Très mauvaise expérience.",
        "Chambre sale et personnel peu aimable.",
        "Déçu par la qualité du service.",
        "Le vol était en retard, très frustrant.",
        "Je ne recommande pas cet hôtel.",
        "Mauvais rapport qualité/prix.",
        "Beaucoup de bruit toute la nuit.",
        "Propreté insuffisante.",
        "Accueil froid et impersonnel.",
        "Trop cher pour les prestations proposées."
    ]

    phrases = {
        1: negative_comments,
        2: negative_comments,
        3: neutral_comments,
        4: positive_comments,
        5: positive_comments
    }
    return random.choice(phrases[note])

def generate_dataset():
    data = []
    for _ in range(1000):
        note = random.randint(1, 5)
        commentaire = generate_comment(note)
        if note <= 2:
            sentiment = 'négatif'
        elif note == 3:
            sentiment = 'neutre'
        else:
            sentiment = 'positif'
        data.append({'note': note, 'commentaire': commentaire, 'sentiment': sentiment})
    return pd.DataFrame(data)

def train_model():
    df = generate_dataset()
    X = df['commentaire']
    y = df['sentiment']
    
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, vectorizer, X_test, y_test

# Entraînement du modèle au démarrage
model, vectorizer, X_test, y_test = train_model()

# Routes HTML (existantes)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyse_sentimentale', methods=['GET', 'POST'])
def analyse_sentimentale():
    if request.method == 'POST':
        comment = request.form['comment']
        
        X_new = vectorizer.transform([comment])
        prediction = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        
        plt.figure(figsize=(8, 6))
        y_test_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_test_pred, labels=['négatif', 'neutre', 'positif'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['négatif', 'neutre', 'positif'],
                    yticklabels=['négatif', 'neutre', 'positif'], cmap='Blues')
        plt.xlabel('Prédit')
        plt.ylabel('Vrai')
        plt.title('Matrice de confusion')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        probabilities = {
            'négatif': round(proba[0] * 100, 2),
            'neutre': round(proba[1] * 100, 2),
            'positif': round(proba[2] * 100, 2)
        }
        
        return render_template('analyse_sentimentale_result.html', 
                             comment=comment,
                             prediction=prediction,
                             probabilities=probabilities,
                             plot_url=plot_url)
    
    return render_template('analyse_sentimentale.html')

@app.route('/serie_temporelle', methods=['GET', 'POST'])
def serie_temporelle():
    if request.method == 'POST':
        try:
            if 'fichier_reservation' not in request.files or 'fichier_temps' not in request.files:
                return render_template('serie_temporelleresult.html', error="Please select both files")
            
            fichier_reservation = request.files['fichier_reservation']
            fichier_temps = request.files['fichier_temps']
            
            if fichier_reservation.filename == '' or fichier_temps.filename == '':
                return render_template('serie_temporelleresult.html', error="No file selected")

            temp_res_path = os.path.join('Fait_Reservations.csv')
            temp_time_path = os.path.join('Dim_Temps.csv')
            
            if os.path.exists(temp_res_path):
                os.remove(temp_res_path)
            if os.path.exists(temp_time_path):
                os.remove(temp_time_path)
                
            fichier_reservation.save(temp_res_path)
            fichier_temps.save(temp_time_path)
            
            encodings_to_try = ['ISO-8859-1', 'utf-8', 'latin1']
            
            df_reservation = None
            for encoding in encodings_to_try:
                try:
                    df_reservation = pd.read_csv(temp_res_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            df_temps = None
            for encoding in encodings_to_try:
                try:
                    df_temps = pd.read_csv(temp_time_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df_reservation is None or df_temps is None:
                return render_template('serie_temporelleresult.html',
                                    error="Failed to read files. Try saving files with UTF-8 encoding.")
            
            print("Aperçu réservation:\n", df_reservation.head(2))
            print("Aperçu temps:\n", df_temps.head(2))
            
            required_cols_res = ['Reservation_PK', 'Temps_FK']
            required_cols_time = ['TempsID', 'Semaine_Année']
            
            def check_columns(df, required_cols):
                df_cols = [col.lower() for col in df.columns]
                missing = []
                for col in required_cols:
                    if col.lower() not in df_cols:
                        missing.append(col)
                return missing
            
            missing_res = check_columns(df_reservation, required_cols_res)
            missing_time = check_columns(df_temps, required_cols_time)
            
            if missing_res:
                return render_template('serie_temporelleresult.html', 
                                    error=f"Reservation file missing columns: {', '.join(missing_res)}. Found: {', '.join(df_reservation.columns)}")
            
            if missing_time:
                return render_template('serie_temporelleresult.html', 
                                    error=f"Time file missing columns: {', '.join(missing_time)}. Found: {', '.join(df_temps.columns)}")
            
            plot_uri, summary = run_forecasting(df_reservation, df_temps)
            
            if plot_uri is None:
                return render_template('serie_temporelleresult.html',
                                    error=summary)
            
            return render_template('serie_temporelleresult.html',
                                plot_uri=plot_uri,
                                summary=summary)
            
        except Exception as e:
            traceback.print_exc()
            return render_template('serie_temporelleresult.html',
                                error=f"System error: {str(e)}")
    
    return render_template('serie_temporelle.html')

@app.route('/classification')
@app.route('/regression')
@app.route('/documentation')
@app.route('/blog')
@app.route('/mentions-legales')
@app.route('/faq')
@app.route('/politique-confidentialite')
@app.route('/conditions-utilisation')
@app.route('/politique-cookies')
@app.route('/contact-juridique')
def redirect_to_regression_vol():
    return redirect('/regression_vol')

@app.route('/regression_vol', methods=['GET', 'POST'])
def regression_vol():
    if request.method == 'POST':
        file = request.files['fichier_vol']
        if file:
            df = pd.read_csv(file, encoding='ISO-8859-1')

            df['DepartHour'] = pd.to_datetime(df['DepartHour'], format='%H:%M')
            df['ArriveHour'] = pd.to_datetime(df['ArriveHour'], format='%H:%M')
            df['FlightDuration'] = (df['ArriveHour'] - df['DepartHour']).dt.total_seconds() / 3600
            df.loc[df['FlightDuration'] < 0, 'FlightDuration'] += 24
            df['DepartHour'] = df['DepartHour'].dt.hour + df['DepartHour'].dt.minute / 60

            df['Airline'] = df['Airline'].astype(str)
            df['DepartAirport'] = df['DepartAirport'].astype(str)
            df['ArriveAirport'] = df['ArriveAirport'].astype(str)

            X = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']]
            X = pd.get_dummies(X, drop_first=True)
            y = df['FlightDuration']

            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            df['Predicted_Duration'] = model.predict(X)

            plt.figure(figsize=(8, 5))
            plt.hist(df['FlightDuration'], bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution des Durées des Vols')
            plt.xlabel('Durée du Vol (en heures)')
            plt.ylabel('Nombre de Vols')

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
            buf.close()

            y_pred = df['Predicted_Duration']
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            result = {
                'MAE': round(mae, 3),
                'MSE': round(mse, 3),
                'R²': round(r2, 3)
            }

            return render_template('regression_result.html', result=result, graph_url=graph_url)

    return render_template('regression_vol.html')

def prepare_data(df_achat, df_reservation, df_temps):
    df_achat = df_achat[df_achat['Client_FK'].notna() & (df_achat['Client_FK'] > 0)]
    df_reservation = df_reservation[df_reservation['ClientID_FK'].notna() & (df_reservation['ClientID_FK'] > 0)]
    
    df_achat['Client_FK'] = df_achat['Client_FK'].astype(int)
    df_reservation['ClientID_FK'] = df_reservation['ClientID_FK'].astype(int)

    achat_stats = df_achat.groupby('Client_FK').agg(
        montant_total_achat=('Prix_Total', 'sum'),
        nb_achats=('AchatID_PK', 'count')
    ).reset_index()

    reservation_stats = df_reservation.groupby('ClientID_FK').agg(
        montant_total_reservation=('Total_Prix', 'sum'),
        nb_reservations=('Reservation_PK', 'count'),
        note_moyenne=('Note_Client', 'mean')
    ).reset_index()

    df_reservation = df_reservation.merge(df_temps, left_on='Temps_FK', right_on='TempsID', how='left')
    voyage_stats = df_reservation.groupby('ClientID_FK').agg(
        nb_jours_voyage=('Date_Complète', 'nunique')
    ).reset_index()

    df_final = achat_stats.merge(reservation_stats, left_on='Client_FK', right_on='ClientID_FK', how='outer')
    df_final = df_final.merge(voyage_stats, on='ClientID_FK', how='left')

    df_final.fillna({
        'montant_total_achat': 0,
        'nb_achats': 0,
        'montant_total_reservation': 0,
        'nb_reservations': 0,
        'note_moyenne': 0,
        'nb_jours_voyage': 0
    }, inplace=True)

    return df_final

def run_clustering(df_final):
    features = ['montant_total_achat', 'nb_achats', 'montant_total_reservation',
                'nb_reservations', 'note_moyenne', 'nb_jours_voyage']
    X = df_final[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_final['cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_final['pca1'] = X_pca[:, 0]
    df_final['pca2'] = X_pca[:, 1]

    silhouette = silhouette_score(X_scaled, df_final['cluster'])
    db_index = davies_bouldin_score(X_scaled, df_final['cluster'])

    return df_final, silhouette, db_index

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'GET':
        return render_template('clustering.html')
    
    if request.method == 'POST':
        if 'fichier_achat' not in request.files or 'fichier_reservation' not in request.files or 'fichier_temps' not in request.files:
            return redirect(request.url)
        
        fichiers = {
            'achat': request.files['fichier_achat'],
            'reservation': request.files['fichier_reservation'],
            'temps': request.files['fichier_temps']
        }
        
        if any(fichier.filename == '' for fichier in fichiers.values()):
            return redirect(request.url)

        try:
            dfs = {}
            for name, fichier in fichiers.items():
                dfs[name] = pd.read_csv(io.StringIO(fichier.read().decode('ISO-8859-1')))
            
            if any(df.empty for df in dfs.values()):
                return render_template('clustering_result.html', error="Un ou plusieurs fichiers sont vides!")
            
            df_final = prepare_data(dfs['achat'], dfs['reservation'], dfs['temps'])
            
            if df_final.empty:
                return render_template('clustering_result.html', error="Aucune donnée valide après préparation!")
            
            df_final, silhouette, db_index = run_clustering(df_final)
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df_final, 
                x='pca1', 
                y='pca2', 
                hue='cluster', 
                palette='Set2',
                s=100,
                alpha=0.7
            )
            plt.title("Visualisation des Clusters (PCA)", fontsize=14)
            plt.xlabel("Composante Principale 1", fontsize=12)
            plt.ylabel("Composante Principale 2", fontsize=12)
            plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
            img.seek(0)
            plot_uri = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close()
            
            return render_template(
                'clustering_result.html',
                plot_uri=plot_uri,
                metrics={
                    'silhouette': round(silhouette, 3),
                    'db_index': round(db_index, 3)
                }
            )
        
        except Exception as e:
            return render_template('clustering_result.html', error=f"Erreur: {str(e)}")

@app.route('/recommandation', methods=['GET', 'POST'])
def recommandation():
    logging.debug("Accessing /recommandation route")
    if request.method == 'POST':
        logging.debug("Received POST request for recommendation")
        try:
            logging.debug(f"Form data: {request.form}")
            logging.debug(f"Files in request: {list(request.files.keys())}")
            
            if 'fichier_achats' not in request.files or 'fichier_produits' not in request.files or 'fichier_clients' not in request.files:
                logging.error("Missing required files")
                return render_template('recommendation_result.html', error="Veuillez sélectionner tous les fichiers nécessaires."), 200
            
            fichier_achats = request.files['fichier_achats']
            fichier_produits = request.files['fichier_produits']
            fichier_clients = request.files['fichier_clients']
            
            logging.debug(f"File names: achats={fichier_achats.filename}, produits={fichier_produits.filename}, clients={fichier_clients.filename}")
            if fichier_achats.filename == '' or fichier_produits.filename == '' or fichier_clients.filename == '':
                logging.error("Empty file names")
                return render_template('recommendation_result.html', error="Aucun fichier sélectionné."), 200
            
            temp_achats_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_achats.filename))
            temp_produits_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_produits.filename))
            temp_clients_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_clients.filename))
            
            logging.debug(f"Saving files to: {temp_achats_path}, {temp_produits_path}, {temp_clients_path}")
            fichier_achats.save(temp_achats_path)
            fichier_produits.save(temp_produits_path)
            fichier_clients.save(temp_clients_path)
            
            for path in [temp_achats_path, temp_produits_path, temp_clients_path]:
                if not os.path.exists(path):
                    logging.error(f"File not found after saving: {path}")
                    raise FileNotFoundError(f"Failed to save file: {path}")
            
            logging.debug("Building recommendation models")
            pivot, model_knn = build_models(temp_achats_path, temp_produits_path, temp_clients_path)
            
            client_nom = request.form.get('client_nom', '').strip().title()
            logging.debug(f"Client name: {client_nom}")
            if not client_nom:
                logging.error("No client name provided")
                return render_template('recommendation_result.html', error="Veuillez entrer un nom de client."), 200
            if client_nom not in pivot.index:
                logging.error(f"Client '{client_nom}' not found in pivot index: {list(pivot.index)}")
                return render_template('recommendation_result.html', error=f"Client '{client_nom}' non trouvé. Clients valides: {', '.join(pivot.index)}"), 200
            
            logging.debug("Getting similar clients")
            clients_similaires = get_clients_similaires(client_nom, pivot, model_knn, n=4)
            
            logging.debug("Generating hybrid recommendations")
            recommandations = score_hybride(client_nom, pivot, clients_similaires, alpha=0.6, top_n=10)
            
            if not recommandations:
                logging.warning("No recommendations generated")
                return render_template('recommendation_result.html', error="Aucune recommandation disponible pour ce client."), 200
            
            logging.debug("Cleaning up temporary files")
            for path in [temp_achats_path, temp_produits_path, temp_clients_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logging.debug("Rendering recommendation results")
            return render_template('recommendation_result.html',
                                 client_nom=client_nom,
                                 recommandations=recommandations,
                                 clients_similaires=clients_similaires), 200
        
        except Exception as e:
            logging.error(f"Error in recommendation route: {str(e)}")
            traceback.print_exc()
            return render_template('recommendation_result.html', error=f"Erreur système : {str(e)}"), 200
    
    logging.debug("Rendering recommendation form for GET request")
    return render_template('recommandation.html'), 200

@app.route('/api/analyse_sentimentale', methods=['POST'])
def api_analyse_sentimentale():
    try:
        comment = request.json.get('comment')
        if not comment:
            return jsonify({'error': 'Missing comment in request'}), 400

        X_new = vectorizer.transform([comment])
        prediction = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        
        plt.figure(figsize=(8, 6))
        y_test_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_test_pred, labels=['négatif', 'neutre', 'positif'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['négatif', 'neutre', 'positif'],
                    yticklabels=['négatif', 'neutre', 'positif'], cmap='Blues')
        plt.xlabel('Prédit')
        plt.ylabel('Vrai')
        plt.title('Matrice de confusion')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        result = {
            'comment': comment,
            'prediction': prediction,
            'probabilities': {
                'négatif': round(proba[0] * 100, 2),
                'neutre': round(proba[1] * 100, 2),
                'positif': round(proba[2] * 100, 2)
            },
            'confusion_matrix': plot_url
        }
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Error in api_analyse_sentimentale: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/serie_temporelle', methods=['POST'])
def api_serie_temporelle():
    try:
        if 'fichier_reservation' not in request.files or 'fichier_temps' not in request.files:
            return jsonify({'error': 'Please select both files'}), 400
        
        fichier_reservation = request.files['fichier_reservation']
        fichier_temps = request.files['fichier_temps']
        
        if fichier_reservation.filename == '' or fichier_temps.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        temp_res_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Fait_Reservations.csv')
        temp_time_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Dim_Temps.csv')
        
        fichier_reservation.save(temp_res_path)
        fichier_temps.save(temp_time_path)

        df_reservation = pd.read_csv(temp_res_path, encoding='ISO-8859-1')
        df_temps = pd.read_csv(temp_time_path, encoding='ISO-8859-1')
        
        plot_uri, summary = run_forecasting(df_reservation, df_temps)
        
        if plot_uri is None:
            return jsonify({'error': summary}), 400

        result = {
            'plot_uri': plot_uri,
            'summary': summary
        }
        os.remove(temp_res_path)
        os.remove(temp_time_path)
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Error in api_serie_temporelle: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/regression_vol', methods=['POST'])
def api_regression_vol():
    try:
        if 'fichier_vol' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['fichier_vol']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_file_path)

        df = pd.read_csv(temp_file_path, encoding='ISO-8859-1')
        df['DepartHour'] = pd.to_datetime(df['DepartHour'], format='%H:%M')
        df['ArriveHour'] = pd.to_datetime(df['ArriveHour'], format='%H:%M')
        df['FlightDuration'] = (df['ArriveHour'] - df['DepartHour']).dt.total_seconds() / 3600
        df.loc[df['FlightDuration'] < 0, 'FlightDuration'] += 24
        df['DepartHour'] = df['DepartHour'].dt.hour + df['DepartHour'].dt.minute / 60

        df['Airline'] = df['Airline'].astype(str)
        df['DepartAirport'] = df['DepartAirport'].astype(str)
        df['ArriveAirport'] = df['ArriveAirport'].astype(str)

        X = pd.get_dummies(df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']], drop_first=True)
        y = df['FlightDuration']

        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        df['Predicted_Duration'] = model.predict(X)

        plt.figure(figsize=(8, 5))
        plt.hist(df['FlightDuration'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution des Durées des Vols')
        plt.xlabel('Durée du Vol (en heures)')
        plt.ylabel('Nombre de Vols')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        y_pred = df['Predicted_Duration']
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Ajouter les prédictions à la réponse
        predictions = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport', 'FlightDuration', 'Predicted_Duration']].to_dict(orient='records')

        result = {
            'graph_url': graph_url,
            'metrics': {
                'MAE': round(mae, 3),
                'MSE': round(mse, 3),
                'R2': round(r2, 3)  # Utiliser "R2" au lieu de "R²" pour compatibilité avec Angular
            },
            'predictions': predictions
        }
        os.remove(temp_file_path)
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Error in api_regression_vol: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clustering', methods=['POST'])
def api_clustering():
    try:
        if 'fichier_achat' not in request.files or 'fichier_reservation' not in request.files or 'fichier_temps' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400
        
        fichiers = {
            'achat': request.files['fichier_achat'],
            'reservation': request.files['fichier_reservation'],
            'temps': request.files['fichier_temps']
        }
        
        if any(fichier.filename == '' for fichier in fichiers.values()):
            return jsonify({'error': 'No file selected'}), 400

        dfs = {name: pd.read_csv(io.StringIO(fichier.read().decode('ISO-8859-1'))) for name, fichier in fichiers.items()}
        
        if any(df.empty for df in dfs.values()):
            return jsonify({'error': 'One or more files are empty'}), 400

        df_final = prepare_data(dfs['achat'], dfs['reservation'], dfs['temps'])
        
        if df_final.empty:
            return jsonify({'error': 'No valid data after preparation'}), 400

        df_final, silhouette, db_index = run_clustering(df_final)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_final, x='pca1', y='pca2', hue='cluster', palette='Set2', s=100, alpha=0.7)
        plt.title("Visualisation des Clusters (PCA)")
        plt.xlabel("Composante Principale 1")
        plt.ylabel("Composante Principale 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
        img.seek(0)
        plot_uri = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        result = {
            'plot_uri': plot_uri,
            'metrics': {
                'silhouette': round(silhouette, 3),
                'db_index': round(db_index, 3)
            }
        }
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Error in api_clustering: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendation', methods=['POST'])
def api_recommendation():
    try:
        if 'fichier_achats' not in request.files or 'fichier_produits' not in request.files or 'fichier_clients' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400
        
        fichier_achats = request.files['fichier_achats']
        fichier_produits = request.files['fichier_produits']
        fichier_clients = request.files['fichier_clients']
        
        if fichier_achats.filename == '' or fichier_produits.filename == '' or fichier_clients.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        temp_achats_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_achats.filename))
        temp_produits_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_produits.filename))
        temp_clients_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fichier_clients.filename))
        
        fichier_achats.save(temp_achats_path)
        fichier_produits.save(temp_produits_path)
        fichier_clients.save(temp_clients_path)

        pivot, model_knn = build_models(temp_achats_path, temp_produits_path, temp_clients_path)
        
        client_nom = request.form.get('client_nom', '').strip().title()
        if not client_nom or client_nom not in pivot.index:
            return jsonify({'error': f"Client '{client_nom}' not found. Valid clients: {', '.join(pivot.index)}"}), 400

        clients_similaires = get_clients_similaires(client_nom, pivot, model_knn, n=4)
        recommandations = score_hybride(client_nom, pivot, clients_similaires, alpha=0.6, top_n=10)

        result = {
            'client_nom': client_nom,
            'recommandations': recommandations,
            'clients_similaires': clients_similaires
        }
        
        for path in [temp_achats_path, temp_produits_path, temp_clients_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Error in api_recommandation: {str(e)}")
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    print("Starting Flask server at http://127.0.0.1:5000", flush=True)
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)