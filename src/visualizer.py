# Module de visualisation des résultats d'entraînement
# Génère des graphiques pour analyser les performances

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os


class Visualizer:
    """
    Classe pour visualiser les résultats de l'entraînement.
    
    Génère des graphiques de:
    - Récompenses par épisode (avec moyenne mobile)
    - Nombre de pommes mangées par épisode
    - Évolution d'epsilon
    - Statistiques globales
    """
    
    @staticmethod
    def plot_training_results(
        metrics: Dict[str, List],
        summary: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Crée une figure complète avec tous les graphiques de training.
        
        Args:
            metrics: Dictionnaire avec les métriques (rewards, apples, steps, epsilons)
            summary: Dictionnaire avec le résumé de l'entraînement
            save_path: Chemin pour sauvegarder la figure (optionnel)
            show: Si True, affiche la figure
        """
        rewards = metrics['rewards']
        apples = metrics['apples']
        steps = metrics['steps']
        epsilons = metrics['epsilons']
        
        n_episodes = len(rewards)
        episodes = np.arange(1, n_episodes + 1)
        
        # Créer une figure avec plusieurs subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Résultats d\'entraînement Q-Learning - Snake', fontsize=16, fontweight='bold')
        
        # 1. Récompenses par épisode
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Récompenses')
        
        # Moyenne mobile (fenêtre de 50 épisodes)
        window = min(50, n_episodes // 10)
        if window > 1:
            moving_avg = Visualizer._moving_average(rewards, window)
            ax1.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense')
        ax1.set_title('Récompenses par épisode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pommes mangées par épisode
        ax2 = axes[0, 1]
        ax2.plot(episodes, apples, alpha=0.3, color='green', label='Pommes')
        
        if window > 1:
            moving_avg_apples = Visualizer._moving_average(apples, window)
            ax2.plot(episodes[window-1:], moving_avg_apples, color='darkgreen', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Nombre de pommes')
        ax2.set_title('Pommes mangées par épisode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Nombre de pas par épisode
        ax3 = axes[1, 0]
        ax3.plot(episodes, steps, alpha=0.3, color='orange', label='Pas')
        
        if window > 1:
            moving_avg_steps = Visualizer._moving_average(steps, window)
            ax3.plot(episodes[window-1:], moving_avg_steps, color='darkorange', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Nombre de pas')
        ax3.set_title('Nombre de pas par épisode')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Évolution d'epsilon
        ax4 = axes[1, 1]
        ax4.plot(episodes, epsilons, color='purple', linewidth=2)
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Évolution du taux d\'exploration (ε)')
        ax4.grid(True, alpha=0.3)
        
        # Ajouter les statistiques dans le titre
        stats_text = (
            f"Récompense moy: {summary['avg_reward']:.2f} | "
            f"Pommes moy: {summary['avg_apples']:.2f} | "
            f"Temps: {summary['training_time']:.1f}s | "
            f"Q-table: {summary['q_table_size']} états"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Graphiques sauvegardés: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_comparison(
        metrics_list: List[Dict[str, List]],
        labels: List[str],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Compare plusieurs entraînements sur un même graphique.
        
        Args:
            metrics_list: Liste de dictionnaires de métriques
            labels: Liste de labels pour chaque entraînement
            save_path: Chemin pour sauvegarder la figure
            show: Si True, affiche la figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Comparaison des entraînements', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
            color = colors[i % len(colors)]
            rewards = metrics['rewards']
            apples = metrics['apples']
            episodes = np.arange(1, len(rewards) + 1)
            
            # Moyenne mobile
            window = min(50, len(rewards) // 10)
            if window > 1:
                moving_avg_rewards = Visualizer._moving_average(rewards, window)
                moving_avg_apples = Visualizer._moving_average(apples, window)
                
                axes[0].plot(episodes[window-1:], moving_avg_rewards, color=color, linewidth=2, label=label)
                axes[1].plot(episodes[window-1:], moving_avg_apples, color=color, linewidth=2, label=label)
        
        axes[0].set_xlabel('Épisode')
        axes[0].set_ylabel('Récompense (moyenne mobile)')
        axes[0].set_title('Récompenses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Épisode')
        axes[1].set_ylabel('Pommes (moyenne mobile)')
        axes[1].set_title('Pommes mangées')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_evaluation_results(
        eval_results: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualise les résultats d'évaluation.
        
        Args:
            eval_results: Dictionnaire avec les résultats d'évaluation
            save_path: Chemin pour sauvegarder la figure
            show: Si True, affiche la figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Résultats d\'évaluation (sans exploration)', fontsize=14, fontweight='bold')
        
        # Récompenses
        axes[0].bar(range(len(eval_results['rewards'])), eval_results['rewards'], color='blue', alpha=0.7)
        axes[0].axhline(eval_results['avg_reward'], color='red', linestyle='--', label='Moyenne')
        axes[0].set_xlabel('Épisode')
        axes[0].set_ylabel('Récompense')
        axes[0].set_title('Récompenses par épisode')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pommes
        axes[1].bar(range(len(eval_results['apples'])), eval_results['apples'], color='green', alpha=0.7)
        axes[1].axhline(eval_results['avg_apples'], color='red', linestyle='--', label='Moyenne')
        axes[1].set_xlabel('Épisode')
        axes[1].set_ylabel('Pommes')
        axes[1].set_title('Pommes mangées par épisode')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Pas
        axes[2].bar(range(len(eval_results['steps'])), eval_results['steps'], color='orange', alpha=0.7)
        axes[2].axhline(eval_results['avg_steps'], color='red', linestyle='--', label='Moyenne')
        axes[2].set_xlabel('Épisode')
        axes[2].set_ylabel('Pas')
        axes[2].set_title('Nombre de pas par épisode')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        """
        Calcule la moyenne mobile.
        
        Args:
            data: Liste de données
            window: Taille de la fenêtre
            
        Returns:
            Array numpy avec les moyennes mobiles
        """
        if window < 1:
            return np.array(data)
        
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    @staticmethod
    def print_summary(summary: Dict):
        """
        Affiche un résumé textuel des résultats.
        
        Args:
            summary: Dictionnaire avec le résumé de l'entraînement
        """
        print("\n" + "="*60)
        print("📈 RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("="*60)
        print(f"Nombre d'épisodes:           {summary['n_episodes']}")
        print(f"Temps d'entraînement:        {summary['training_time']:.1f}s")
        print(f"Taille de la Q-table:        {summary['q_table_size']} états")
        print("-"*60)
        print(f"Récompense moyenne:          {summary['avg_reward']:.2f}")
        print(f"Récompense (100 derniers):   {summary['avg_reward_last_100']:.2f}")
        print(f"Récompense max:              {summary['max_reward']:.2f}")
        print(f"Récompense min:              {summary['min_reward']:.2f}")
        print("-"*60)
        print(f"Pommes moyennes:             {summary['avg_apples']:.2f}")
        print(f"Pommes (100 derniers):       {summary['avg_apples_last_100']:.2f}")
        print(f"Pommes max:                  {summary['max_apples']}")
        print(f"Pommes totales:              {summary['total_apples']}")
        print("-"*60)
        print(f"Pas moyens:                  {summary['avg_steps']:.0f}")
        print(f"Epsilon final:               {summary['final_epsilon']:.4f}")
        print("="*60 + "\n")
