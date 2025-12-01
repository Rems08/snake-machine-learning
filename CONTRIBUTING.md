# Contribution au projet

Merci de votre intérêt pour contribuer au projet Snake Q-Learning !

## Comment contribuer

### Signaler un bug

Si vous trouvez un bug :

1. Vérifiez qu'il n'a pas déjà été signalé dans les [Issues](https://github.com/Rems08/snake-machine-learning/issues)
2. Créez une nouvelle issue avec :
   - Description claire du problème
   - Étapes pour reproduire
   - Comportement attendu vs comportement actuel
   - Version de Python et dépendances

### Proposer une amélioration

Pour proposer une nouvelle fonctionnalité :

1. Créez une issue pour en discuter
2. Attendez les retours avant de commencer l'implémentation
3. Suivez les directives de code

### Soumettre une Pull Request

1. **Fork le repository**
   ```bash
   git clone https://github.com/Rems08/snake-machine-learning.git
   cd snake-machine-learning
   ```

2. **Créer une branche**
   ```bash
   git checkout -b feature/ma-fonctionnalite
   # ou
   git checkout -b fix/mon-bug
   ```

3. **Installer l'environnement de développement**
   ```bash
   uv sync
   ```

4. **Faire vos modifications**
   - Suivez le style de code existant
   - Ajoutez des tests si applicable
   - Mettez à jour la documentation

5. **Tester vos modifications**
   ```bash
   uv run python src/test_installation.py
   ```

6. **Commit et push**
   ```bash
   git add .
   git commit -m "feat: description de la fonctionnalité"
   git push origin feature/ma-fonctionnalite
   ```

7. **Créer une Pull Request**
   - Décrivez clairement vos changements
   - Référencez les issues liées
   - Attendez la review

## Conventions de code

### Style Python

- Suivre [PEP 8](https://peps.python.org/pep-0008/)
- Utiliser des noms de variables explicites
- Ajouter des docstrings aux fonctions/classes
- Limiter les lignes à 100 caractères

### Commits

Suivre la convention [Conventional Commits](https://www.conventionalcommits.org/) :

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation
- `style:` formatage
- `refactor:` refactoring
- `test:` ajout de tests
- `chore:` maintenance

Exemples :
```
feat: ajouter support pour grilles rectangulaires
fix: corriger collision avec le corps du serpent
docs: améliorer README avec exemples
```

## Structure du code

```
src/
├── snake_environment.py    # Environnement du jeu (ne pas casser l'API)
├── q_learning_agent.py     # Agent (garder compatible)
├── trainer.py              # Entraînement
├── app.py                  # Interface Streamlit
└── test_installation.py    # Tests
```

### Ajouter une fonctionnalité

Exemple : Ajouter un nouveau type de récompense

1. **Modifier l'environnement**
   ```python
   # Dans snake_environment.py
   def step(self, action):
       # ... code existant ...
       
       # Nouvelle récompense
       if self.nouvelle_condition():
           reward += self.nouvelle_recompense
       
       return state, reward, done, info
   ```

2. **Documenter**
   ```python
   """
   Récompenses:
   - +10 : manger pomme
   - -10 : mourir
   - -0.1 : par pas
   - +5 : nouvelle récompense (description)
   """
   ```

3. **Tester**
   ```python
   # Ajouter un test dans test_installation.py
   def test_nouvelle_fonctionnalite():
       env = SnakeEnvironment()
       # ... test ...
   ```

## Idées de contribution

### Faciles (débutants)

- Améliorer la documentation
- Ajouter des exemples d'utilisation
- Corriger des typos
- Améliorer les messages d'erreur

### Moyennes

- Ajouter de nouvelles visualisations
- Implémenter des variantes du jeu
- Améliorer l'interface Streamlit
- Ajouter des tests unitaires

### Avancées

- Implémenter Deep Q-Learning (DQN)
- Ajouter d'autres algorithmes RL
- Optimiser les performances
- Créer un mode multijoueur

## Tests

Avant de soumettre :

```bash
# Tests basiques
uv run python src/test_installation.py

# Test de l'interface
uv run streamlit run src/app.py
```

## Documentation

Lors de l'ajout de fonctionnalités :

1. Mettre à jour `README.md`
2. Ajouter dans `MODELISATION.md` si pertinent
3. Documenter dans le code (docstrings)
4. Ajouter des exemples

## Questions ?

N'hésitez pas à :
- Créer une issue pour discuter
- Demander des clarifications
- Proposer des améliorations

Merci de votre contribution ! 🎉
