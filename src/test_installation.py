"""
Script de test rapide pour vérifier l'installation et l'entraînement.
"""

from snake_environment import SnakeEnvironment
from q_learning_agent import QLearningAgent
from trainer import train_snake_agent


def test_environment():
    """Test de l'environnement Snake."""
    print("Test de l'environnement Snake...")
    env = SnakeEnvironment(grid_size=5, max_steps=50)
    state = env.reset()
    print(f"  ✓ État initial: {state}")
    print(f"  ✓ Grille:\n{env.render()}")
    
    # Effectuer quelques actions
    for i in range(5):
        action = i % 4
        state, reward, done, info = env.step(action)
        if done:
            break
    
    print(f"  ✓ Après 5 actions: Score = {info['score']}, Steps = {info['steps']}")
    print("  ✓ Test environnement réussi!\n")


def test_agent():
    """Test de l'agent Q-Learning."""
    print("Test de l'agent Q-Learning...")
    agent = QLearningAgent()
    
    # État fictif
    state = (0, 1, 0, 0, 1, 0, 0, 0)
    action = agent.choose_action(state)
    print(f"  ✓ Action choisie pour état {state}: {action}")
    
    # Mise à jour
    next_state = (1, 0, 1, 0, 0, 1, 0, 0)
    agent.update(state, action, 10.0, next_state, False)
    print(f"  ✓ Mise à jour effectuée")
    print(f"  ✓ Stats agent: {agent.get_stats()}")
    print("  ✓ Test agent réussi!\n")


def test_quick_training():
    """Test d'un entraînement rapide."""
    print("Test d'un entraînement rapide (10 épisodes)...")
    results = train_snake_agent(
        grid_size=5,
        max_steps=30,
        n_episodes=10,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9,
        seed=42,
        verbose=False
    )
    
    print(f"  ✓ Entraînement terminé!")
    print(f"  ✓ Récompense moyenne: {results['final_stats']['mean_reward_last_100']:.2f}")
    print(f"  ✓ Pommes moyennes: {results['final_stats']['mean_apples_last_100']:.2f}")
    print(f"  ✓ États explorés: {results['final_stats']['q_table_size']}")
    print(f"  ✓ Agent sauvegardé: {results['agent_path']}")
    print("  ✓ Test entraînement réussi!\n")


def main():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("TESTS DU PROJET SNAKE Q-LEARNING")
    print("="*60 + "\n")
    
    try:
        test_environment()
        test_agent()
        test_quick_training()
        
        print("="*60)
        print("✅ TOUS LES TESTS SONT RÉUSSIS!")
        print("="*60)
        print("\nVous pouvez maintenant lancer l'application web:")
        print("  streamlit run src/app.py")
        print()
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
