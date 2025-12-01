"""
Agent Q-Learning pour le jeu Snake.
"""

import pickle
from typing import Tuple, Optional
from collections import defaultdict
import random


class QLearningAgent:
    """Agent qui apprend à jouer à Snake avec Q-Learning."""
    
    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            n_actions: Nombre d'actions possibles
            alpha: Taux d'apprentissage (learning rate)
            gamma: Facteur de discount pour les récompenses futures
            epsilon: Taux d'exploration initial
            epsilon_min: Taux d'exploration minimum
            epsilon_decay: Facteur de décroissance de epsilon
            seed: Graine pour la génération aléatoire
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        
        # Table Q : Q[state][action] = valeur
        self.q_table = defaultdict(lambda: [0.0] * n_actions)
        
        # Métriques
        self.total_updates = 0
    
    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        Choisit une action selon la politique epsilon-greedy.
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon prend la meilleure action
            
        Returns:
            Action choisie (0-3)
        """
        # Exploration : action aléatoire
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        
        # Exploitation : meilleure action selon Q
        q_values = self.q_table[state]
        max_q = max(q_values)
        
        # Si plusieurs actions ont la même valeur max, en choisir une au hasard
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return self.rng.choice(best_actions)
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Met à jour la table Q avec la règle de Q-Learning.
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: Nouvel état
            done: Episode terminé
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Si l'épisode est terminé, pas de valeur future
            target = reward
        else:
            # Sinon, ajouter la valeur future maximale
            max_next_q = max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        # Mise à jour Q-Learning
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
        self.total_updates += 1
    
    def decay_epsilon(self):
        """Réduit le taux d'exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """
        Sauvegarde l'agent (table Q et paramètres).
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        data = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'n_actions': self.n_actions,
            'total_updates': self.total_updates
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """
        Charge un agent sauvegardé.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions, data['q_table'])
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.n_actions = data['n_actions']
        self.total_updates = data.get('total_updates', 0)
    
    def get_stats(self) -> dict:
        """
        Retourne des statistiques sur l'agent.
        
        Returns:
            Dictionnaire de statistiques
        """
        return {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_updates': self.total_updates,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
