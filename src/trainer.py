# Module de training pour l'agent Q-Learning
# Gère l'entraînement et le suivi des métriques

from typing import Dict, List, Tuple
import time
from snake_env import SnakeEnvironment
from q_learning_agent import QLearningAgent


class Trainer:
    """
    Classe pour entraîner l'agent Q-Learning sur l'environnement Snake.
    
    Suit les métriques importantes:
    - Récompenses par épisode
    - Nombre de pommes mangées par épisode
    - Nombre de pas par épisode
    - Taux d'exploration (epsilon)
    """
    
    def __init__(self, env: SnakeEnvironment, agent: QLearningAgent):
        """
        Initialise le trainer.
        
        Args:
            env: L'environnement Snake
            agent: L'agent Q-Learning
        """
        self.env = env
        self.agent = agent
        
        # Métriques d'entraînement
        self.episode_rewards: List[float] = []
        self.episode_apples: List[int] = []
        self.episode_steps: List[int] = []
        self.episode_epsilons: List[float] = []
        
        # Informations d'entraînement
        self.training_time: float = 0.0
        self.is_trained: bool = False
    
    def train(
        self,
        n_episodes: int = 1000,
        max_steps: int = 200,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict:
        """
        Entraîne l'agent sur un nombre d'épisodes.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum de pas par épisode
            verbose: Si True, affiche les progrès
            log_interval: Intervalle d'affichage des logs
            
        Returns:
            Dictionnaire avec les statistiques d'entraînement
        """
        start_time = time.time()
        
        if verbose:
            print(f"🎯 Début de l'entraînement: {n_episodes} épisodes")
            print(f"   Paramètres: α={self.agent.learning_rate}, γ={self.agent.discount_factor}")
            print(f"   Epsilon: {self.agent.epsilon:.3f} → {self.agent.epsilon_min:.3f}")
            print("-" * 60)
        
        for episode in range(n_episodes):
            # Réinitialiser l'environnement
            state = self.env.reset()
            episode_reward = 0.0
            
            for step in range(max_steps):
                # Choisir et exécuter une action
                action = self.agent.get_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                # Mettre à jour l'agent
                self.agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Enregistrer les métriques
            self.episode_rewards.append(episode_reward)
            self.episode_apples.append(self.env.get_score())
            self.episode_steps.append(self.env.get_steps())
            self.episode_epsilons.append(self.agent.epsilon)
            
            # Décrémenter epsilon
            self.agent.decay_epsilon()
            
            # Afficher les progrès
            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = sum(self.episode_rewards[-log_interval:]) / log_interval
                avg_apples = sum(self.episode_apples[-log_interval:]) / log_interval
                avg_steps = sum(self.episode_steps[-log_interval:]) / log_interval
                
                print(f"Épisode {episode + 1}/{n_episodes}")
                print(f"  Récompense moy: {avg_reward:6.2f}")
                print(f"  Pommes moy:     {avg_apples:6.2f}")
                print(f"  Pas moy:        {avg_steps:6.0f}")
                print(f"  Epsilon:        {self.agent.epsilon:6.3f}")
                print(f"  Q-table:        {self.agent.get_q_table_size()} états")
                print("-" * 60)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            print(f"\n✅ Entraînement terminé en {self.training_time:.1f}s")
            print(f"   Q-table finale: {self.agent.get_q_table_size()} états explorés")
        
        return self.get_training_summary()
    
    def get_training_summary(self) -> Dict:
        """
        Retourne un résumé de l'entraînement.
        
        Returns:
            Dictionnaire avec les statistiques globales
        """
        if not self.episode_rewards:
            return {}
        
        n_episodes = len(self.episode_rewards)
        
        # Calculer les moyennes sur différentes périodes
        last_100 = min(100, n_episodes)
        
        return {
            'n_episodes': n_episodes,
            'training_time': self.training_time,
            'total_reward': sum(self.episode_rewards),
            'avg_reward': sum(self.episode_rewards) / n_episodes,
            'avg_reward_last_100': sum(self.episode_rewards[-last_100:]) / last_100,
            'max_reward': max(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'total_apples': sum(self.episode_apples),
            'avg_apples': sum(self.episode_apples) / n_episodes,
            'avg_apples_last_100': sum(self.episode_apples[-last_100:]) / last_100,
            'max_apples': max(self.episode_apples),
            'avg_steps': sum(self.episode_steps) / n_episodes,
            'q_table_size': self.agent.get_q_table_size(),
            'final_epsilon': self.agent.epsilon,
            'agent_parameters': self.agent.get_parameters()
        }
    
    def get_metrics(self) -> Dict[str, List]:
        """
        Retourne toutes les métriques d'entraînement.
        
        Returns:
            Dictionnaire avec les listes de métriques par épisode
        """
        return {
            'rewards': self.episode_rewards,
            'apples': self.episode_apples,
            'steps': self.episode_steps,
            'epsilons': self.episode_epsilons
        }
    
    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps: int = 200,
        verbose: bool = True
    ) -> Dict:
        """
        Évalue l'agent entraîné (sans exploration).
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre maximum de pas par épisode
            verbose: Si True, affiche les résultats
            
        Returns:
            Dictionnaire avec les statistiques d'évaluation
        """
        eval_rewards = []
        eval_apples = []
        eval_steps = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            
            for step in range(max_steps):
                # Action greedy (pas d'exploration)
                action = self.agent.get_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_apples.append(self.env.get_score())
            eval_steps.append(self.env.get_steps())
        
        results = {
            'avg_reward': sum(eval_rewards) / n_episodes,
            'avg_apples': sum(eval_apples) / n_episodes,
            'avg_steps': sum(eval_steps) / n_episodes,
            'max_apples': max(eval_apples),
            'min_apples': min(eval_apples),
            'rewards': eval_rewards,
            'apples': eval_apples,
            'steps': eval_steps
        }
        
        if verbose:
            print(f"\n📊 Évaluation sur {n_episodes} épisodes:")
            print(f"   Récompense moyenne: {results['avg_reward']:.2f}")
            print(f"   Pommes moyennes:    {results['avg_apples']:.2f}")
            print(f"   Pommes max:         {results['max_apples']}")
            print(f"   Pas moyens:         {results['avg_steps']:.0f}")
        
        return results
    
    def reset_metrics(self):
        """Réinitialise toutes les métriques d'entraînement."""
        self.episode_rewards = []
        self.episode_apples = []
        self.episode_steps = []
        self.episode_epsilons = []
        self.training_time = 0.0
        self.is_trained = False
