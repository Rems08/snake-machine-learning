"""
Script d'entraînement pour l'agent Snake.
"""

import json
import os
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime

from snake_environment import SnakeEnvironment
from q_learning_agent import QLearningAgent


class Trainer:
    """Classe pour gérer l'entraînement de l'agent."""
    
    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 200,
        n_episodes: int = 1000,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        save_dir: str = 'models'
    ):
        """
        Initialise le trainer.
        
        Args:
            grid_size: Taille de la grille
            max_steps: Nombre maximum de pas par épisode
            n_episodes: Nombre d'épisodes d'entraînement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Taux d'exploration initial
            epsilon_min: Taux d'exploration minimum
            epsilon_decay: Facteur de décroissance de epsilon
            seed: Graine aléatoire
            save_dir: Répertoire de sauvegarde
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed
        self.save_dir = save_dir
        
        # Créer le répertoire de sauvegarde
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialiser l'environnement et l'agent
        self.env = SnakeEnvironment(grid_size=grid_size, max_steps=max_steps, seed=seed)
        self.agent = QLearningAgent(
            n_actions=4,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed
        )
        
        # Métriques d'entraînement
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_apples: List[int] = []
        self.episode_epsilons: List[float] = []
        
        # Paramètres d'entraînement
        self.params = {
            'grid_size': grid_size,
            'max_steps': max_steps,
            'n_episodes': n_episodes,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon_initial': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'seed': seed
        }
    
    def train(self, verbose: bool = True, save_freq: int = 100) -> Dict:
        """
        Entraîne l'agent.
        
        Args:
            verbose: Afficher les informations pendant l'entraînement
            save_freq: Fréquence de sauvegarde (en épisodes)
            
        Returns:
            Dictionnaire avec les résultats d'entraînement
        """
        if verbose:
            print(f"Début de l'entraînement pour {self.n_episodes} épisodes...")
            print(f"Paramètres: α={self.agent.alpha}, γ={self.agent.gamma}, "
                  f"ε={self.agent.epsilon:.2f}→{self.agent.epsilon_min}")
        
        for episode in range(self.n_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            # Boucle de l'épisode
            while not self.env.done:
                # Choisir et exécuter une action
                action = self.agent.choose_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                # Mettre à jour l'agent
                self.agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
            
            # Décroisser epsilon
            self.agent.decay_epsilon()
            
            # Sauvegarder les métriques
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.env.steps)
            self.episode_apples.append(self.env.apples_eaten)
            self.episode_epsilons.append(self.agent.epsilon)
            
            # Affichage périodique
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_apples = np.mean(self.episode_apples[-100:])
                print(f"Épisode {episode + 1}/{self.n_episodes} - "
                      f"Récompense moy.: {avg_reward:.2f} - "
                      f"Pommes moy.: {avg_apples:.2f} - "
                      f"ε: {self.agent.epsilon:.3f} - "
                      f"États: {len(self.agent.q_table)}")
            
            # Sauvegarde périodique
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(episode + 1)
        
        # Sauvegarde finale
        results = self._save_final_results()
        
        if verbose:
            print("\nEntraînement terminé!")
            print(f"Récompense moyenne finale (100 derniers épisodes): "
                  f"{np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"Pommes moyennes (100 derniers épisodes): "
                  f"{np.mean(self.episode_apples[-100:]):.2f}")
            print(f"États explorés: {len(self.agent.q_table)}")
        
        return results
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict:
        """
        Évalue l'agent entraîné.
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            render: Afficher le jeu
            
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        eval_rewards = []
        eval_apples = []
        eval_lengths = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            if render:
                print(f"\n=== Épisode {episode + 1} ===")
                print(self.env.render())
            
            while not self.env.done:
                # Choisir la meilleure action (pas d'exploration)
                action = self.agent.choose_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if render:
                    print(f"\nAction: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
                    print(self.env.render())
            
            eval_rewards.append(episode_reward)
            eval_apples.append(self.env.apples_eaten)
            eval_lengths.append(self.env.steps)
            
            if render:
                print(f"\nRécompense: {episode_reward}, Pommes: {self.env.apples_eaten}")
        
        return {
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'mean_apples': float(np.mean(eval_apples)),
            'std_apples': float(np.std(eval_apples)),
            'mean_length': float(np.mean(eval_lengths)),
            'std_length': float(np.std(eval_lengths))
        }
    
    def _save_checkpoint(self, episode: int):
        """Sauvegarde un checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{episode}.pkl')
        self.agent.save(checkpoint_path)
    
    def _save_final_results(self) -> Dict:
        """Sauvegarde les résultats finaux."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarder l'agent
        agent_path = os.path.join(self.save_dir, f'agent_{timestamp}.pkl')
        self.agent.save(agent_path)
        
        # Préparer les résultats
        results = {
            'timestamp': timestamp,
            'params': self.params,
            'agent_path': agent_path,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_apples': self.episode_apples,
            'episode_epsilons': self.episode_epsilons,
            'final_stats': {
                'mean_reward_last_100': float(np.mean(self.episode_rewards[-100:])),
                'mean_apples_last_100': float(np.mean(self.episode_apples[-100:])),
                'total_episodes': len(self.episode_rewards),
                'q_table_size': len(self.agent.q_table),
                'total_updates': self.agent.total_updates
            }
        }
        
        # Sauvegarder les résultats en JSON
        results_path = os.path.join(self.save_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['results_path'] = results_path
        
        return results
    
    def load_agent(self, filepath: str):
        """Charge un agent sauvegardé."""
        self.agent.load(filepath)


def train_snake_agent(
    grid_size: int = 10,
    max_steps: int = 200,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = None,
    save_dir: str = 'models',
    verbose: bool = True
) -> Dict:
    """
    Fonction principale d'entraînement.
    
    Args:
        grid_size: Taille de la grille
        max_steps: Nombre maximum de pas par épisode
        n_episodes: Nombre d'épisodes d'entraînement
        alpha: Taux d'apprentissage
        gamma: Facteur de discount
        epsilon: Taux d'exploration initial
        epsilon_min: Taux d'exploration minimum
        epsilon_decay: Facteur de décroissance de epsilon
        seed: Graine aléatoire
        save_dir: Répertoire de sauvegarde
        verbose: Afficher les informations
        
    Returns:
        Résultats d'entraînement
    """
    trainer = Trainer(
        grid_size=grid_size,
        max_steps=max_steps,
        n_episodes=n_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
        save_dir=save_dir
    )
    
    results = trainer.train(verbose=verbose)
    
    return results


if __name__ == '__main__':
    # Exemple d'utilisation
    results = train_snake_agent(
        grid_size=10,
        max_steps=200,
        n_episodes=1000,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=42,
        verbose=True
    )
    
    print(f"\nRésultats sauvegardés dans: {results['results_path']}")
    print(f"Agent sauvegardé dans: {results['agent_path']}")
