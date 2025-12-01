#!/usr/bin/env python3
"""
Script de test rapide pour vérifier que tout fonctionne.
Lance un entraînement court (100 épisodes) et affiche les résultats.
"""

from snake_env import SnakeEnvironment
from q_learning_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer


def test_environment():
    """Test de l'environnement Snake."""
    print("🧪 Test de l'environnement...")
    env = SnakeEnvironment(width=20, height=20)
    
    # Test reset
    state = env.reset()
    print(f"   État initial: {state}")
    print(f"   Forme: {len(state)} caractéristiques")
    
    # Test step
    for _ in range(10):
        action = 0  # UP
        state, reward, done = env.step(action)
        if done:
            break
    
    print("   ✅ Environnement OK")


def test_agent():
    """Test de l'agent Q-Learning."""
    print("\n🤖 Test de l'agent...")
    agent = QLearningAgent(n_actions=4)
    
    # Test get_action
    state = (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1)
    action = agent.get_action(state)
    print(f"   Action choisie: {action}")
    
    # Test update
    next_state = (0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0)
    agent.update(state, action, 10.0, next_state, False)
    print(f"   Q-value mise à jour: {agent.q_table[state][action]:.3f}")
    
    print("   ✅ Agent OK")


def test_training():
    """Test d'un entraînement court."""
    print("\n🎓 Test d'entraînement (100 épisodes)...")
    
    env = SnakeEnvironment(width=20, height=20)
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.1
    )
    
    trainer = Trainer(env, agent)
    trainer.train(n_episodes=100, max_steps=100, verbose=False)
    
    summary = trainer.get_training_summary()
    print(f"\n   Résultats:")
    print(f"   - Récompense moyenne: {summary['avg_reward']:.2f}")
    print(f"   - Pommes moyennes: {summary['avg_apples']:.2f}")
    print(f"   - Q-table: {summary['q_table_size']} états")
    print(f"   - Temps: {summary['training_time']:.1f}s")
    
    print("   ✅ Training OK")


def test_save_load():
    """Test de sauvegarde et chargement."""
    print("\n💾 Test de sauvegarde/chargement...")
    
    # Créer et entraîner un agent
    env = SnakeEnvironment(width=20, height=20)
    agent = QLearningAgent()
    trainer = Trainer(env, agent)
    trainer.train(n_episodes=50, max_steps=50, verbose=False)
    
    # Sauvegarder
    agent.save("test_agent.pkl")
    q_table_size_before = agent.get_q_table_size()
    print(f"   Agent sauvegardé: {q_table_size_before} états")
    
    # Charger
    new_agent = QLearningAgent()
    new_agent.load("test_agent.pkl")
    q_table_size_after = new_agent.get_q_table_size()
    print(f"   Agent chargé: {q_table_size_after} états")
    
    assert q_table_size_before == q_table_size_after
    print("   ✅ Sauvegarde/Chargement OK")


def main():
    """Lance tous les tests."""
    print("=" * 60)
    print("🧪 TESTS DU SYSTÈME SNAKE Q-LEARNING")
    print("=" * 60)
    
    try:
        test_environment()
        test_agent()
        test_training()
        test_save_load()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS SONT PASSÉS!")
        print("=" * 60)
        print("\n💡 Le système est prêt à être utilisé:")
        print("   • Interface: python main.py")
        print("   • CLI: python train_cli.py")
        print("   • Aide: python train_cli.py --help")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
