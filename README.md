# Snake avec Q-Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UV](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://github.com/astral-sh/uv)

Projet d'apprentissage par renforcement : un agent apprend à jouer au jeu Snake en utilisant l'algorithme Q-Learning.

## 🎯 Objectif

Implémenter une version du jeu Snake dans laquelle un agent apprend automatiquement à :
- Se déplacer sur une grille
- Manger des pommes
- Éviter les murs et son propre corps
- Maximiser sa récompense cumulée

## 🧠 Modélisation RL

### États
L'état est représenté par un tuple de 8 éléments :
- **Direction du serpent** (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
- **Danger devant** (booléen)
- **Danger à gauche** (booléen)
- **Danger à droite** (booléen)
- **Pomme devant** (booléen)
- **Pomme à gauche** (booléen)
- **Pomme à droite** (booléen)
- **Pomme derrière** (booléen)

### Actions
4 actions possibles :
- 0 : UP (⬆️)
- 1 : DOWN (⬇️)
- 2 : LEFT (⬅️)
- 3 : RIGHT (➡️)

Le demi-tour immédiat est bloqué pour éviter les collisions instantanées.

### Récompenses
- **+10** : Le serpent mange une pomme
- **-10** : Le serpent meurt (collision avec mur ou corps)
- **-0.1** : À chaque pas (encourage l'efficacité)

## 🚀 Installation

### Avec UV (recommandé)

```bash
# Installer UV si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Créer l'environnement et installer les dépendances
uv sync

# Activer l'environnement
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### Avec pip

```bash
pip install -r requirements.txt
```

## 🎮 Utilisation

### Interface Web (Streamlit)

Lancer l'application web :

```bash
streamlit run src/app.py
```

L'interface propose deux onglets :

#### 🎓 Training
- Configurer les paramètres d'entraînement (taille grille, épisodes, α, γ, ε)
- Lancer l'entraînement avec visualisation de la progression
- Sauvegarder automatiquement l'agent et les résultats

#### 📊 Résultats
- Visualiser les courbes d'apprentissage (récompenses, pommes mangées, epsilon)
- Afficher les statistiques finales
- Rejouer une partie avec l'agent entraîné

### Entraînement en ligne de commande

```bash
python src/trainer.py
```

## 📁 Structure du projet

```
snake-machine-learning/
├── src/
│   ├── snake_environment.py    # Environnement du jeu Snake
│   ├── q_learning_agent.py     # Agent Q-Learning
│   ├── trainer.py              # Script d'entraînement
│   └── app.py                  # Interface web Streamlit
├── models/                     # Agents et résultats sauvegardés
├── pyproject.toml             # Configuration UV
├── requirements.txt           # Dépendances pip
└── README.md
```

## 🧪 Paramètres recommandés

Pour un bon apprentissage :
- **Grille** : 10x10
- **Épisodes** : 1000-2000
- **Alpha (α)** : 0.1
- **Gamma (γ)** : 0.9
- **Epsilon initial** : 1.0
- **Epsilon min** : 0.01
- **Epsilon decay** : 0.995

## 📊 Résultats attendus

Après ~1000 épisodes, l'agent devrait :
- Manger en moyenne 3-5 pommes par partie
- Éviter efficacement les obstacles
- Développer des stratégies de déplacement intelligentes

## 🛠️ Technologies utilisées

- **Python 3.9+**
- **Streamlit** : Interface web
- **NumPy** : Calculs numériques
- **Matplotlib** : Visualisation
- **Plotly** : Graphiques interactifs

## 📝 Licence

Ce projet est réalisé dans un cadre éducatif.

## 👨‍💻 Auteur

Projet réalisé dans le cadre du cours d'apprentissage par renforcement.
