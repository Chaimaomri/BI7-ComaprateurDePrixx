 [BI7-ComparateurDePrix] - [Comparaison de prix par ML] - [PowerBi-Dashboarding]

## 🎯 Objectif
"Comparer automatiquement les prix de produits, les réservations d'hôtels et les vols en utilisant le Machine Learning" 

Ce projet consiste à développer une solution de Business Intelligence (BI) associée à des fonctionnalités avancées de Machine Learning (ML) et d'Intelligence Artificielle (IA) pour l'analyse, la comparaison et la prévision des prix des produits proposés par différents vendeurs. L'objectif est de fournir aux clients des outils d'analyse interactifs et aux entreprises des mécanismes pour optimiser leurs stratégies commerciales.

## 🛠️ Installation
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/ChaimaOmri/BI7-ComaprateurDePrixx.git

.
├── data/               # Données CSV des produits
├── notebooks/          # Analyses exploratoires (.ipynb)
├── src/
│   ├── preprocessing/  # Scripts de nettoyage
│   └── training/       # Scripts d'entraînement
└── models/             # Modèles sauvegardés




## 📊 Interface Power BI : **CheckNCompare**

Ce rapport Power BI interactif est destiné à **trois profils utilisateurs** :

| 🧑‍💼 Profil                   | Objectif |
|-----------------------------|----------|
| **Sales Director**           | Analyse du CA, diversité produits, billets par destination |
| **Marketing Director**       | Comparaison promo/non-promo, marques top ventes, satisfaction |
| **Chief Financial Officer**  | Flux de trésorerie, revenu mensuel, profit, dépenses vs revenus |

### ✨ Fonctionnalités incluses :
- Navigation par onglets / rôles
- Visualisations claires : barres, secteurs, cartes
- Filtres dynamiques par date
- **Modèles ML intégrés** :
  - Prédictions de réservation
  - Segmentation client
  - Analyse de sentiment client

---

## 🚀 Déploiement Power BI

### 1. Extraction automatique du modèle (via [pbi-tools](https://pbi.tools))
```bash
pbi-tools extract "pwbinour.pbix" watch


.
├── data/ # Données CSV brutes
├── notebooks/ # Analyses exploratoires et ML (Jupyter)
├── src/
│ ├── preprocessing/ # Scripts de nettoyage des données
│ └── training/ # Scripts d'entraînement de modèles ML
├── models/ # Modèles ML sauvegardés
├── pwbinour.pbix # Rapport Power BI complet (visuel)
├── pwbinour/ # Modèle Power BI extrait (scripts DAX/M via pbi-tools)
│ └── Model/ # TMDL : DAX, relations, tables
│ └── Connections.json # Scripts Power Query (M)
├── README.md # Présentation complète du projet
