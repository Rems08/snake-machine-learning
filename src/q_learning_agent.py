# Agent Q-Learning pour Snake
# Implémentation de l'algorithme Q-Learning avec exploration epsilon-greedy

import random
import pickle
from typing import Dict, Tuple, Optional
from collections import defaultdict


class QLearningAgent:
    """
    Agent Q-Learning pour apprendre à jouer au Snake.
    
    Utilise une table Q (dictionnaire) pour stocker les valeurs Q(s, a).
    Stratégie d'exploration: epsilon-greedy.
    """
    
    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            n_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur d'escompte (gamma)
            epsilon: Probabilité d'exploration initiale
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Table Q : dictionnaire avec valeurs par défaut à 0
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: {a: 0.0 for a in range(n_actions)})
        
        # Statistiques
        self.training_episodes = 0
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Choisit une action selon la stratégie epsilon-greedy.
        
        Args:
            state: L'état actuel
            training: Si True, utilise epsilon-greedy; sinon, utilise la politique greedy
            
        Returns:
            L'action choisie
        """
        # Exploration : action aléatoire
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Exploitation : action avec la meilleure valeur Q
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        
        # Si plusieurs actions ont la même valeur Q maximale, en choisir une au hasard
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Met à jour la valeur Q selon l'équation de Q-Learning.
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: L'état actuel
            action: L'action effectuée
            reward: La récompense reçue
            next_state: Le nouvel état
            done: True si l'épisode est terminé
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Si l'épisode est terminé, pas de future récompense
            target_q = reward
        else:
            # Meilleure action future
            max_next_q = max(self.q_table[next_state].values())
            target_q = reward + self.discount_factor * max_next_q
        
        # Mise à jour de Q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Décroît epsilon après chaque épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1
    
    def save(self, filepath: str):
        """
        Sauvegarde l'agent (Q-table et paramètres).
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        data = {
            'q_table': dict(self.q_table),  # Convertir defaultdict en dict
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'training_episodes': self.training_episodes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """
        Charge un agent sauvegardé.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruire la Q-table
            self.q_table = defaultdict(lambda: {a: 0.0 for a in range(self.n_actions)})
            for state, actions in data['q_table'].items():
                self.q_table[state] = actions
            
            # Charger les paramètres
            self.n_actions = data['n_actions']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.epsilon = data['epsilon']
            self.epsilon_decay = data['epsilon_decay']
            self.epsilon_min = data['epsilon_min']
            self.training_episodes = data['training_episodes']
            
            return True
        except FileNotFoundError:
            return False
    
    def get_q_table_size(self) -> int:
        """Retourne le nombre d'entrées dans la Q-table."""
        return len(self.q_table)
    
    def get_parameters(self) -> Dict:
        """Retourne les paramètres de l'agent."""
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'training_episodes': self.training_episodes,
            'q_table_size': self.get_q_table_size()
        }
