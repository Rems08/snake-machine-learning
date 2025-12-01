#!/usr/bin/env python3
"""
Script CLI pour entraîner l'agent Q-Learning Snake sans interface graphique.
Idéal pour des entraînements rapides ou sur des machines sans affichage.
"""

import argparse
import os
from snake_env import SnakeEnvironment
from q_learning_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer


def main():
    """Point d'entrée principal du script CLI."""
    
    parser = argparse.ArgumentParser(
        description="Entraîner un agent Q-Learning à jouer au Snake",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paramètres d'entraînement
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1000,
        help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=200,
        help="Nombre maximum de pas par épisode"
    )
    
    # Hyperparamètres Q-Learning
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.1,
        help="Taux d'apprentissage (learning rate)"
    )
    parser.add_argument(
        "--gamma", "-g",
        type=float,
        default=0.95,
        help="Facteur d'escompte (discount factor)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Probabilité d'exploration initiale"
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Facteur de décroissance d'epsilon"
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Valeur minimale d'epsilon"
    )
    
    # Paramètres environnement
    parser.add_argument(
        "--width",
        type=int,
        default=40,
        help="Largeur de la grille"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=40,
        help="Hauteur de la grille"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed pour la reproductibilité"
    )
    
    # Options d'affichage
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les détails d'entraînement"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Intervalle d'affichage des logs (en épisodes)"
    )
    
    # Sauvegarde et visualisation
    parser.add_argument(
        "--save-agent",
        type=str,
        default="snake_agent.pkl",
        help="Chemin de sauvegarde de l'agent"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="training_results.png",
        help="Chemin de sauvegarde des graphiques"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Ne pas générer de graphiques"
    )
    
    # Évaluation
    parser.add_argument(
        "--evaluate",
        type=int,
        default=0,
        help="Nombre d'épisodes d'évaluation après l'entraînement (0 = pas d'évaluation)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐍 SNAKE Q-LEARNING - ENTRAÎNEMENT CLI")
    print("=" * 70)
    
    # Créer l'environnement
    print("\n📦 Initialisation de l'environnement...")
    env = SnakeEnvironment(
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    print(f"   Grille: {args.width}x{args.height}")
    
    # Créer l'agent
    print("\n🤖 Création de l'agent Q-Learning...")
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min
    )
    print(f"   α (alpha):      {args.alpha}")
    print(f"   γ (gamma):      {args.gamma}")
    print(f"   ε (epsilon):    {args.epsilon} → {args.epsilon_min}")
    print(f"   decay:          {args.epsilon_decay}")
    
    # Créer le trainer
    trainer = Trainer(env, agent)
    
    # Entraîner
    print("\n🎓 DÉBUT DE L'ENTRAÎNEMENT")
    print("-" * 70)
    
    trainer.train(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose,
        log_interval=args.log_interval
    )
    
    # Résumé
    summary = trainer.get_training_summary()
    Visualizer.print_summary(summary)
    
    # Sauvegarder l'agent
    print(f"\n💾 Sauvegarde de l'agent: {args.save_agent}")
    agent.save(args.save_agent)
    
    # Générer les graphiques
    if not args.no_plot:
        print(f"\n📊 Génération des graphiques: {args.save_plot}")
        metrics = trainer.get_metrics()
        Visualizer.plot_training_results(
            metrics,
            summary,
            save_path=args.save_plot,
            show=False
        )
    
    # Évaluation
    if args.evaluate > 0:
        print(f"\n🎯 ÉVALUATION ({args.evaluate} épisodes)")
        print("-" * 70)
        eval_results = trainer.evaluate(
            n_episodes=args.evaluate,
            max_steps=args.max_steps,
            verbose=True
        )
        
        if not args.no_plot:
            eval_plot_path = args.save_plot.replace(".png", "_eval.png")
            print(f"\n📊 Génération des graphiques d'évaluation: {eval_plot_path}")
            Visualizer.plot_evaluation_results(
                eval_results,
                save_path=eval_plot_path,
                show=False
            )
    
    print("\n✅ TERMINÉ!")
    print("=" * 70)
    
    # Afficher les commandes pour visualiser
    print("\n📋 Prochaines étapes:")
    print(f"   • Voir les graphiques: open {args.save_plot}")
    print(f"   • Charger l'agent: agent.load('{args.save_agent}')")
    if args.evaluate > 0:
        print(f"   • Voir l'évaluation: open {args.save_plot.replace('.png', '_eval.png')}")
    print()


if __name__ == "__main__":
    main()
