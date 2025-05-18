import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO

from google.colab import files
uploaded = files.upload()

# Charger le fichier CSV
df = pd.read_csv('offres_tunisair.csv', sep=';', encoding='ISO-8859-1')


# Fonction pour déterminer la classe
def get_price_class(price):
    if price < 500000:
        return "Classe_0"
    elif price <= 800000:
        return "Classe_1"
    else:
        return "Classe_2"

def clean_price(p):
    try:
        p = p.replace(" TND", "").replace(",", ".").strip()
        return float(p)
    except:
        return None  # ou np.nan si tu veux filtrer avec .dropna()


# Créer les dossiers si non existants
os.makedirs("images/Classe_0", exist_ok=True)
os.makedirs("images/Classe_1", exist_ok=True)
os.makedirs("images/Classe_2", exist_ok=True)

# Appliquer nettoyage
df['Prix_nettoye'] = df['Prix'].apply(clean_price)


# Boucle de téléchargement d’images
for index, row in df.iterrows():
    try:
        prix = clean_price(row['Prix'])
        if prix is None:
            raise ValueError("Prix invalide")
        classe = get_price_class(prix)
        url = row['Image']
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img.save(f"images/{classe}/{index}.jpg")
    except Exception as e:
        print(f"Erreur pour l'image {index} : {e}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



# Résumé du modèle
model.summary()

import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Créer une copie nettoyée du DataFrame original
df_clean = df[['Prix', 'Prix_nettoye']].copy()
df_clean['Classe_reelle'] = df_clean['Prix_nettoye'].apply(get_price_class)
df_clean['Index'] = df_clean.index.astype(str)
df_clean['Chemin_image'] = df_clean.apply(lambda row: f"images/{row['Classe_reelle']}/{row.name}.jpg", axis=1)

# Fonction de prédiction sur une seule image
def predict_class(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = model.predict(x, verbose=0)
        classe_predite = np.argmax(preds, axis=1)[0]
        return f"Classe_{classe_predite}"
    except Exception as e:
        print(f"Erreur avec {image_path} : {e}")
        return "Erreur"


# Prédire les classes
df_clean['Classe_predite'] = df_clean['Chemin_image'].apply(predict_class)

# Nettoyer les lignes où la prédiction a échoué
df_resultats = df_clean[df_clean['Classe_predite'] != "Erreur"].copy()

# Réorganiser les colonnes
df_resultats_final = df_resultats[['Index', 'Prix', 'Prix_nettoye', 'Classe_reelle', 'Classe_predite', 'Chemin_image']]

# Exporter dans un fichier CSV lisible
df_resultats_final.to_csv('resultats_predictions_tunisair_nettoye.csv', index=False, encoding='utf-8-sig', sep=';')

print("✅ Fichier CSV propre enregistré : resultats_predictions_tunisair_nettoye.csv")
