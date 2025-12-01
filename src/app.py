"""
Interface web pour le jeu Snake avec apprentissage par renforcement.
Utilise Streamlit pour créer une interface avec onglets Training et Résultats.
"""

import streamlit as st
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from snake_environment import SnakeEnvironment
from q_learning_agent import QLearningAgent
from trainer import Trainer


# Configuration de la page
st.set_page_config(
    page_title="Snake Q-Learning",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def load_results():
    """Charge tous les fichiers de résultats disponibles."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return []
    
    results_files = [f for f in os.listdir(models_dir) if f.startswith('results_') and f.endswith('.json')]
    results = []
    
    for filename in sorted(results_files, reverse=True):
        filepath = os.path.join(models_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['filename'] = filename
                results.append(data)
        except Exception as e:
            st.warning(f"Erreur lors du chargement de {filename}: {e}")
    
    return results


def plot_training_curves(results):
    """Affiche les courbes d'entraînement."""
    episode_rewards = results['episode_rewards']
    episode_apples = results['episode_apples']
    episode_epsilons = results['episode_epsilons']
    
    # Calculer les moyennes glissantes
    window = 50
    rewards_smooth = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    apples_smooth = np.convolve(episode_apples, np.ones(window)/window, mode='valid')
    
    # Créer les graphiques avec Plotly
    fig = go.Figure()
    
    # Récompenses
    fig.add_trace(go.Scatter(
        y=episode_rewards,
        mode='lines',
        name='Récompense',
        line=dict(color='lightblue', width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        y=rewards_smooth,
        mode='lines',
        name=f'Récompense (moyenne {window} ép.)',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Récompense par épisode',
        xaxis_title='Épisode',
        yaxis_title='Récompense',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pommes mangées
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=episode_apples,
        mode='lines',
        name='Pommes',
        line=dict(color='lightgreen', width=1),
        opacity=0.3
    ))
    fig2.add_trace(go.Scatter(
        y=apples_smooth,
        mode='lines',
        name=f'Pommes (moyenne {window} ép.)',
        line=dict(color='green', width=2)
    ))
    
    fig2.update_layout(
        title='Nombre de pommes mangées par épisode',
        xaxis_title='Épisode',
        yaxis_title='Pommes mangées',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Epsilon
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        y=episode_epsilons,
        mode='lines',
        name='Epsilon',
        line=dict(color='orange', width=2)
    ))
    
    fig3.update_layout(
        title='Taux d\'exploration (epsilon) au cours de l\'entraînement',
        xaxis_title='Épisode',
        yaxis_title='Epsilon',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)


def render_snake_game(grid_state, score, steps, snake_length):
    """Affiche l'état actuel du jeu Snake."""
    # Créer une représentation colorée de la grille
    colors = ['white', 'green', 'darkgreen', 'red']  # Vide, corps, tête, pomme
    grid_size = len(grid_state)
    
    # Créer une figure matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Créer une image RGB
    img = np.zeros((grid_size, grid_size, 3))
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_state[i][j] == 0:  # Vide
                img[i, j] = [1, 1, 1]
            elif grid_state[i][j] == 1:  # Corps
                img[i, j] = [0.2, 0.8, 0.2]
            elif grid_state[i][j] == 2:  # Tête
                img[i, j] = [0, 0.5, 0]
            elif grid_state[i][j] == 3:  # Pomme
                img[i, j] = [1, 0, 0]
    
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Score: {score} | Steps: {steps} | Longueur: {snake_length}')
    
    # Ajouter une grille
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    st.pyplot(fig)
    plt.close()


def play_demo(agent_path, grid_size, max_steps):
    """Joue une partie de démonstration avec l'agent entraîné."""
    # Charger l'agent
    agent = QLearningAgent()
    agent.load(agent_path)
    
    # Créer l'environnement
    env = SnakeEnvironment(grid_size=grid_size, max_steps=max_steps)
    state = env.reset()
    
    # Placeholder pour l'affichage
    game_placeholder = st.empty()
    info_placeholder = st.empty()
    
    episode_reward = 0
    
    while not env.done:
        # Choisir la meilleure action
        action = agent.choose_action(state, training=False)
        
        # Afficher l'état actuel
        with game_placeholder.container():
            render_snake_game(env.get_grid_state(), env.score, env.steps, len(env.snake))
        
        with info_placeholder.container():
            st.write(f"**Action:** {['⬆️ UP', '⬇️ DOWN', '⬅️ LEFT', '➡️ RIGHT'][action]}")
        
        # Exécuter l'action
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        time.sleep(0.3)  # Pause pour l'animation
    
    # Afficher l'état final
    with game_placeholder.container():
        render_snake_game(env.get_grid_state(), env.score, env.steps, len(env.snake))
    
    with info_placeholder.container():
        st.success(f"**Partie terminée!**")
        st.write(f"**Récompense totale:** {episode_reward:.2f}")
        st.write(f"**Pommes mangées:** {env.apples_eaten}")
        st.write(f"**Nombre de pas:** {env.steps}")


# Interface principale
def main():
    st.title("🐍 Snake avec Q-Learning")
    st.markdown("---")
    
    # Onglets
    tab1, tab2 = st.tabs(["🎓 Training", "📊 Résultats"])
    
    # ===== ONGLET TRAINING =====
    with tab1:
        st.header("Configuration de l'entraînement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paramètres de l'environnement")
            grid_size = st.slider("Taille de la grille", 5, 20, 10)
            max_steps = st.slider("Nombre max de pas par épisode", 100, 500, 200, step=50)
            seed = st.number_input("Graine aléatoire (seed)", 0, 9999, 42, step=1)
        
        with col2:
            st.subheader("Paramètres de l'agent")
            n_episodes = st.slider("Nombre d'épisodes", 100, 5000, 1000, step=100)
            alpha = st.slider("Alpha (α) - Taux d'apprentissage", 0.01, 1.0, 0.1, step=0.01)
            gamma = st.slider("Gamma (γ) - Facteur de discount", 0.0, 1.0, 0.9, step=0.05)
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Paramètres d'exploration")
            epsilon = st.slider("Epsilon initial (ε)", 0.1, 1.0, 1.0, step=0.1)
            epsilon_min = st.slider("Epsilon minimum", 0.001, 0.5, 0.01, step=0.001)
        
        with col4:
            st.subheader(" ")
            st.write("")  # Espaceur
            epsilon_decay = st.slider("Epsilon decay", 0.9, 0.999, 0.995, step=0.001)
        
        st.markdown("---")
        
        # Bouton d'entraînement
        if st.button("🚀 Lancer l'entraînement", type="primary"):
            with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                # Créer une barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Créer le trainer
                trainer = Trainer(
                    grid_size=grid_size,
                    max_steps=max_steps,
                    n_episodes=n_episodes,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    seed=seed
                )
                
                # Entraîner avec mise à jour de la progression
                for episode in range(n_episodes):
                    state = trainer.env.reset()
                    episode_reward = 0
                    
                    while not trainer.env.done:
                        action = trainer.agent.choose_action(state, training=True)
                        next_state, reward, done, info = trainer.env.step(action)
                        trainer.agent.update(state, action, reward, next_state, done)
                        episode_reward += reward
                        state = next_state
                    
                    trainer.agent.decay_epsilon()
                    trainer.episode_rewards.append(episode_reward)
                    trainer.episode_lengths.append(trainer.env.steps)
                    trainer.episode_apples.append(trainer.env.apples_eaten)
                    trainer.episode_epsilons.append(trainer.agent.epsilon)
                    
                    # Mise à jour de la progression
                    if (episode + 1) % max(1, n_episodes // 100) == 0:
                        progress = (episode + 1) / n_episodes
                        progress_bar.progress(progress)
                        avg_reward = np.mean(trainer.episode_rewards[-100:])
                        avg_apples = np.mean(trainer.episode_apples[-100:])
                        status_text.text(f"Épisode {episode + 1}/{n_episodes} - "
                                       f"Récompense: {avg_reward:.2f} - "
                                       f"Pommes: {avg_apples:.2f}")
                
                # Sauvegarder les résultats
                results = trainer._save_final_results()
                
                progress_bar.progress(1.0)
                status_text.text("Entraînement terminé!")
            
            st.success("✅ Entraînement terminé avec succès!")
            
            # Afficher les statistiques finales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Récompense moyenne (100 derniers)", 
                         f"{results['final_stats']['mean_reward_last_100']:.2f}")
            with col2:
                st.metric("Pommes moyennes (100 derniers)", 
                         f"{results['final_stats']['mean_apples_last_100']:.2f}")
            with col3:
                st.metric("États explorés", 
                         results['final_stats']['q_table_size'])
            with col4:
                st.metric("Mises à jour totales", 
                         results['final_stats']['total_updates'])
            
            st.info(f"💾 Résultats sauvegardés: `{results['results_path']}`")
            st.info(f"🤖 Agent sauvegardé: `{results['agent_path']}`")
    
    # ===== ONGLET RÉSULTATS =====
    with tab2:
        st.header("Résultats des entraînements")
        
        # Charger les résultats disponibles
        results_list = load_results()
        
        if not results_list:
            st.warning("⚠️ Aucun résultat d'entraînement disponible. "
                      "Lancez d'abord un entraînement dans l'onglet Training.")
        else:
            # Sélectionner un résultat
            result_options = [f"{r['timestamp']} - {r['params']['n_episodes']} épisodes" 
                            for r in results_list]
            selected_idx = st.selectbox("Sélectionner un entraînement", 
                                       range(len(result_options)),
                                       format_func=lambda x: result_options[x])
            
            selected_results = results_list[selected_idx]
            
            # Afficher les paramètres
            st.subheader("Paramètres de l'entraînement")
            params = selected_results['params']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Grille:** {params['grid_size']}x{params['grid_size']}")
                st.write(f"**Max steps:** {params['max_steps']}")
            with col2:
                st.write(f"**Épisodes:** {params['n_episodes']}")
                st.write(f"**Alpha (α):** {params['alpha']}")
            with col3:
                st.write(f"**Gamma (γ):** {params['gamma']}")
                st.write(f"**Epsilon initial:** {params['epsilon_initial']}")
            with col4:
                st.write(f"**Epsilon min:** {params['epsilon_min']}")
                st.write(f"**Epsilon decay:** {params['epsilon_decay']}")
            
            st.markdown("---")
            
            # Statistiques finales
            st.subheader("Statistiques finales")
            stats = selected_results['final_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Récompense moyenne (100 derniers)", 
                         f"{stats['mean_reward_last_100']:.2f}")
            with col2:
                st.metric("Pommes moyennes (100 derniers)", 
                         f"{stats['mean_apples_last_100']:.2f}")
            with col3:
                st.metric("États explorés", stats['q_table_size'])
            with col4:
                st.metric("Mises à jour totales", stats['total_updates'])
            
            st.markdown("---")
            
            # Courbes d'apprentissage
            st.subheader("Courbes d'apprentissage")
            plot_training_curves(selected_results)
            
            st.markdown("---")
            
            # Démonstration
            st.subheader("🎮 Démonstration de l'agent entraîné")
            st.write("Regardez l'agent jouer avec la politique apprise!")
            
            if st.button("▶️ Rejouer une partie", type="primary"):
                play_demo(
                    selected_results['agent_path'],
                    params['grid_size'],
                    params['max_steps']
                )


if __name__ == '__main__':
    main()
