# Snake avec apprentissage par renforcement (Q-Learning)
# Interface Pyxel avec menus de training et visualisation

from collections import deque, namedtuple
import pyxel
import os
import threading

from snake_env import SnakeEnvironment, ACTION_TO_DIRECTION, UP, DOWN, LEFT, RIGHT
from q_learning_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer

Point = namedtuple("Point", ["x", "y"])

# États de l'application
STATE_MENU = 0
STATE_PLAY = 1
STATE_TRAINING_CONFIG = 2
STATE_TRAINING = 3
STATE_RESULTS = 4
STATE_DEMO = 5

# Constantes graphiques
COL_BACKGROUND = 3
COL_BODY = 11
COL_HEAD = 7
COL_DEATH = 8
COL_APPLE = 8
COL_TEXT = 7
COL_MENU_BG = 1
COL_MENU_SELECTED = 10

WIDTH = 40
HEIGHT = 50
HEIGHT_SCORE = pyxel.FONT_HEIGHT


class SnakeRL:
    """Application principale Snake avec Q-Learning."""
    
    def __init__(self):
        """Initialise l'application Pyxel."""
        pyxel.init(WIDTH, HEIGHT, title="Snake RL - Q-Learning", fps=20, display_scale=12)
        define_sound_and_music()
        
        # État de l'application
        self.app_state = STATE_MENU
        self.menu_selection = 0
        self.menu_options = ["JOUER (Humain)", "TRAINING IA", "RÉSULTATS", "DEMO IA", "QUITTER"]
        
        # Configuration de training
        self.config = {
            'n_episodes': 1000,
            'max_steps': 200,
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
        self.config_param = 0  # Paramètre actuellement édité
        self.config_params = ['n_episodes', 'max_steps', 'learning_rate', 
                              'discount_factor', 'epsilon', 'epsilon_decay', 'epsilon_min']
        
        # Environnement et agent RL
        self.env = None
        self.agent = None
        self.trainer = None
        self.training_thread = None
        self.training_done = False
        self.training_progress = ""
        
        # Jeu humain (mode classique)
        self.reset_game()
        
        # Mode démo
        self.demo_speed = 10  # FPS pour la démo
        self.demo_frame_counter = 0
        
        pyxel.run(self.update, self.draw)
    
    def reset_game(self):
        """Réinitialise le jeu pour le mode humain."""
        self.direction = ACTION_TO_DIRECTION[RIGHT]
        self.snake = deque()
        self.snake.append(Point(20, 25))
        self.death = False
        self.score = 0
        self.generate_apple()
    
    def generate_apple(self):
        """Génère une pomme aléatoire."""
        snake_pixels = set(self.snake)
        self.apple = self.snake[0]
        while self.apple in snake_pixels:
            x = pyxel.rndi(0, WIDTH - 1)
            y = pyxel.rndi(HEIGHT_SCORE + 1, HEIGHT - 1)
            self.apple = Point(x, y)
    
    ############
    # UPDATE #
    ############
    
    def update(self):
        """Mise à jour principale selon l'état."""
        if self.app_state == STATE_MENU:
            self.update_menu()
        elif self.app_state == STATE_PLAY:
            self.update_play()
        elif self.app_state == STATE_TRAINING_CONFIG:
            self.update_training_config()
        elif self.app_state == STATE_TRAINING:
            self.update_training()
        elif self.app_state == STATE_RESULTS:
            self.update_results()
        elif self.app_state == STATE_DEMO:
            self.update_demo()
    
    def update_menu(self):
        """Mise à jour du menu principal."""
        if pyxel.btnp(pyxel.KEY_UP):
            self.menu_selection = (self.menu_selection - 1) % len(self.menu_options)
        elif pyxel.btnp(pyxel.KEY_DOWN):
            self.menu_selection = (self.menu_selection + 1) % len(self.menu_options)
        elif pyxel.btnp(pyxel.KEY_RETURN) or pyxel.btnp(pyxel.KEY_SPACE):
            if self.menu_selection == 0:  # Jouer
                self.reset_game()
                self.app_state = STATE_PLAY
            elif self.menu_selection == 1:  # Training
                self.app_state = STATE_TRAINING_CONFIG
            elif self.menu_selection == 2:  # Résultats
                if self.trainer and self.trainer.is_trained:
                    self.app_state = STATE_RESULTS
            elif self.menu_selection == 3:  # Démo
                if self.agent and self.trainer and self.trainer.is_trained:
                    self.start_demo()
                    self.app_state = STATE_DEMO
            elif self.menu_selection == 4:  # Quitter
                pyxel.quit()
    
    def update_play(self):
        """Mise à jour du mode jeu humain."""
        if not self.death:
            self.update_direction()
            self.update_snake()
            self.check_death()
            self.check_apple()
        
        if pyxel.btnp(pyxel.KEY_Q) or pyxel.btnp(pyxel.KEY_ESCAPE):
            self.app_state = STATE_MENU
        
        if pyxel.btnp(pyxel.KEY_R):
            self.reset_game()
    
    def update_direction(self):
        """Gère les contrôles du joueur."""
        if pyxel.btn(pyxel.KEY_UP):
            if self.direction != ACTION_TO_DIRECTION[DOWN]:
                self.direction = ACTION_TO_DIRECTION[UP]
        elif pyxel.btn(pyxel.KEY_DOWN):
            if self.direction != ACTION_TO_DIRECTION[UP]:
                self.direction = ACTION_TO_DIRECTION[DOWN]
        elif pyxel.btn(pyxel.KEY_LEFT):
            if self.direction != ACTION_TO_DIRECTION[RIGHT]:
                self.direction = ACTION_TO_DIRECTION[LEFT]
        elif pyxel.btn(pyxel.KEY_RIGHT):
            if self.direction != ACTION_TO_DIRECTION[LEFT]:
                self.direction = ACTION_TO_DIRECTION[RIGHT]
    
    def update_snake(self):
        """Déplace le serpent."""
        old_head = self.snake[0]
        new_head = Point(old_head.x + self.direction.x, old_head.y + self.direction.y)
        self.snake.appendleft(new_head)
        self.popped_point = self.snake.pop()
    
    def check_apple(self):
        """Vérifie si le serpent mange une pomme."""
        if self.snake[0] == self.apple:
            self.score += 1
            self.snake.append(self.popped_point)
            self.generate_apple()
            pyxel.play(0, 0)
    
    def check_death(self):
        """Vérifie si le serpent meurt."""
        head = self.snake[0]
        if head.x < 0 or head.y < HEIGHT_SCORE or head.x >= WIDTH or head.y >= HEIGHT:
            self.death_event()
        elif len(self.snake) != len(set(self.snake)):
            self.death_event()
    
    def death_event(self):
        """Gère la mort du serpent."""
        self.death = True
        pyxel.play(0, 1)
    
    def update_training_config(self):
        """Mise à jour de la configuration de training."""
        # Navigation
        if pyxel.btnp(pyxel.KEY_UP):
            self.config_param = (self.config_param - 1) % len(self.config_params)
        elif pyxel.btnp(pyxel.KEY_DOWN):
            self.config_param = (self.config_param + 1) % len(self.config_params)
        
        # Modification des valeurs
        param_name = self.config_params[self.config_param]
        if pyxel.btnp(pyxel.KEY_LEFT):
            self.adjust_config(param_name, -1)
        elif pyxel.btnp(pyxel.KEY_RIGHT):
            self.adjust_config(param_name, 1)
        
        # Lancer le training
        if pyxel.btnp(pyxel.KEY_RETURN) or pyxel.btnp(pyxel.KEY_SPACE):
            self.start_training()
            self.app_state = STATE_TRAINING
        
        # Retour
        if pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
            self.app_state = STATE_MENU
    
    def adjust_config(self, param_name, direction):
        """Ajuste un paramètre de configuration."""
        if param_name == 'n_episodes':
            self.config[param_name] = max(100, self.config[param_name] + direction * 100)
        elif param_name == 'max_steps':
            self.config[param_name] = max(50, self.config[param_name] + direction * 50)
        elif param_name in ['learning_rate', 'discount_factor', 'epsilon', 
                            'epsilon_decay', 'epsilon_min']:
            step = 0.01 if param_name != 'epsilon_decay' else 0.001
            self.config[param_name] = max(0.0, min(1.0, self.config[param_name] + direction * step))
    
    def start_training(self):
        """Lance l'entraînement dans un thread séparé."""
        self.training_done = False
        self.training_progress = "Initialisation..."
        
        # Créer l'environnement et l'agent
        self.env = SnakeEnvironment(width=WIDTH, height=HEIGHT - HEIGHT_SCORE)
        self.agent = QLearningAgent(
            n_actions=4,
            learning_rate=self.config['learning_rate'],
            discount_factor=self.config['discount_factor'],
            epsilon=self.config['epsilon'],
            epsilon_decay=self.config['epsilon_decay'],
            epsilon_min=self.config['epsilon_min']
        )
        self.trainer = Trainer(self.env, self.agent)
        
        # Lancer le training dans un thread
        def train():
            self.training_progress = "Training en cours..."
            self.trainer.train(
                n_episodes=self.config['n_episodes'],
                max_steps=self.config['max_steps'],
                verbose=False
            )
            self.training_progress = "Training terminé!"
            self.training_done = True
            
            # Sauvegarder l'agent
            self.agent.save("snake_agent.pkl")
            
            # Générer les graphiques
            metrics = self.trainer.get_metrics()
            summary = self.trainer.get_training_summary()
            Visualizer.plot_training_results(metrics, summary, save_path="training_results.png", show=False)
        
        self.training_thread = threading.Thread(target=train)
        self.training_thread.start()
    
    def update_training(self):
        """Mise à jour pendant le training."""
        if self.training_done:
            if pyxel.btnp(pyxel.KEY_RETURN) or pyxel.btnp(pyxel.KEY_SPACE):
                self.app_state = STATE_RESULTS
            elif pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
                self.app_state = STATE_MENU
    
    def update_results(self):
        """Mise à jour de l'écran des résultats."""
        if pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
            self.app_state = STATE_MENU
        elif pyxel.btnp(pyxel.KEY_D):
            self.start_demo()
            self.app_state = STATE_DEMO
    
    def start_demo(self):
        """Initialise une partie en mode démo."""
        if self.env:
            self.demo_state = self.env.reset()
            self.demo_done = False
            self.demo_frame_counter = 0
    
    def update_demo(self):
        """Mise à jour du mode démo (IA joue)."""
        if pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
            self.app_state = STATE_MENU
            return
        
        if not self.demo_done:
            self.demo_frame_counter += 1
            
            # Contrôler la vitesse de la démo
            if self.demo_frame_counter % (20 // self.demo_speed) == 0:
                action = self.agent.get_action(self.demo_state, training=False)
                self.demo_state, reward, self.demo_done = self.env.step(action)
        else:
            # Recommencer
            if pyxel.btnp(pyxel.KEY_SPACE) or pyxel.btnp(pyxel.KEY_RETURN):
                self.start_demo()
    
    ##########
    # DRAW #
    ##########
    
    def draw(self):
        """Rendu principal selon l'état."""
        if self.app_state == STATE_MENU:
            self.draw_menu()
        elif self.app_state == STATE_PLAY:
            self.draw_play()
        elif self.app_state == STATE_TRAINING_CONFIG:
            self.draw_training_config()
        elif self.app_state == STATE_TRAINING:
            self.draw_training()
        elif self.app_state == STATE_RESULTS:
            self.draw_results()
        elif self.app_state == STATE_DEMO:
            self.draw_demo()
    
    def draw_menu(self):
        """Dessine le menu principal."""
        pyxel.cls(COL_MENU_BG)
        
        # Titre
        title = "SNAKE RL"
        self.draw_centered_text(title, 10, COL_TEXT)
        subtitle = "Q-Learning"
        self.draw_centered_text(subtitle, 18, COL_TEXT)
        
        # Options
        for i, option in enumerate(self.menu_options):
            y = 30 + i * 8
            color = COL_MENU_SELECTED if i == self.menu_selection else COL_TEXT
            
            # Indicateur de disponibilité
            available = True
            if i == 2 and (not self.trainer or not self.trainer.is_trained):
                available = False
                option += " (non dispo)"
            elif i == 3 and (not self.agent or not self.trainer or not self.trainer.is_trained):
                available = False
                option += " (non dispo)"
            
            if not available:
                color = 5  # Gris
            
            self.draw_centered_text(option, y, color)
        
        # Instructions
        self.draw_centered_text("UP/DOWN: Naviguer", HEIGHT - 15, 6)
        self.draw_centered_text("ENTER: Sélectionner", HEIGHT - 8, 6)
    
    def draw_play(self):
        """Dessine le mode jeu humain."""
        if not self.death:
            pyxel.cls(col=COL_BACKGROUND)
            self.draw_snake(self.snake)
            self.draw_score_bar(self.score)
            pyxel.pset(self.apple.x, self.apple.y, col=COL_APPLE)
        else:
            self.draw_death_screen()
    
    def draw_snake(self, snake):
        """Dessine le serpent."""
        for i, point in enumerate(snake):
            color = COL_HEAD if i == 0 else COL_BODY
            pyxel.pset(point.x, point.y, col=color)
    
    def draw_score_bar(self, score):
        """Dessine la barre de score."""
        score_text = f"{score:04}"
        pyxel.rect(0, 0, WIDTH, HEIGHT_SCORE, 5)
        pyxel.text(1, 1, score_text, 6)
    
    def draw_death_screen(self):
        """Dessine l'écran de mort."""
        pyxel.cls(col=COL_DEATH)
        self.draw_centered_text("GAME OVER", 15, 0)
        self.draw_centered_text(f"{self.score:04}", 23, 0)
        self.draw_centered_text("(R)ESTART", 31, 0)
        self.draw_centered_text("(Q)UIT", 39, 0)
    
    def draw_training_config(self):
        """Dessine l'écran de configuration du training."""
        pyxel.cls(COL_MENU_BG)
        
        self.draw_centered_text("CONFIGURATION TRAINING", 5, COL_TEXT)
        
        y = 15
        param_names = {
            'n_episodes': 'Épisodes',
            'max_steps': 'Pas max',
            'learning_rate': 'Alpha (α)',
            'discount_factor': 'Gamma (γ)',
            'epsilon': 'Epsilon (ε)',
            'epsilon_decay': 'Decay ε',
            'epsilon_min': 'Min ε'
        }
        
        for i, param in enumerate(self.config_params):
            color = COL_MENU_SELECTED if i == self.config_param else COL_TEXT
            name = param_names[param]
            value = self.config[param]
            
            if isinstance(value, int):
                text = f"{name}: {value}"
            else:
                text = f"{name}: {value:.3f}"
            
            pyxel.text(5, y, text, color)
            y += 7
        
        self.draw_centered_text("GAUCHE/DROITE: Ajuster", HEIGHT - 20, 6)
        self.draw_centered_text("ENTER: Lancer", HEIGHT - 13, 6)
        self.draw_centered_text("ESC: Retour", HEIGHT - 6, 6)
    
    def draw_training(self):
        """Dessine l'écran de training."""
        pyxel.cls(COL_MENU_BG)
        
        self.draw_centered_text("TRAINING EN COURS", 10, COL_TEXT)
        self.draw_centered_text(self.training_progress, 20, COL_TEXT)
        
        if self.training_done:
            summary = self.trainer.get_training_summary()
            y = 30
            self.draw_centered_text(f"Temps: {summary['training_time']:.1f}s", y, 7)
            y += 8
            self.draw_centered_text(f"Récompense moy: {summary['avg_reward']:.2f}", y, 7)
            y += 8
            self.draw_centered_text(f"Pommes moy: {summary['avg_apples']:.2f}", y, 7)
            y += 8
            self.draw_centered_text(f"Q-table: {summary['q_table_size']} états", y, 7)
            
            self.draw_centered_text("ENTER: Voir résultats", HEIGHT - 13, 6)
            self.draw_centered_text("ESC: Menu", HEIGHT - 6, 6)
    
    def draw_results(self):
        """Dessine l'écran des résultats."""
        pyxel.cls(COL_MENU_BG)
        
        self.draw_centered_text("RÉSULTATS", 5, COL_TEXT)
        
        if self.trainer and self.trainer.is_trained:
            summary = self.trainer.get_training_summary()
            y = 15
            
            texts = [
                f"Épisodes: {summary['n_episodes']}",
                f"Temps: {summary['training_time']:.1f}s",
                f"Récompense moy: {summary['avg_reward']:.2f}",
                f"Pommes moy: {summary['avg_apples']:.2f}",
                f"Pommes max: {summary['max_apples']}",
                f"Q-table: {summary['q_table_size']} états"
            ]
            
            for text in texts:
                pyxel.text(5, y, text, COL_TEXT)
                y += 7
        
        self.draw_centered_text("Graphiques: training_results.png", HEIGHT - 20, 10)
        self.draw_centered_text("(D)EMO: Voir l'IA jouer", HEIGHT - 13, 6)
        self.draw_centered_text("ESC: Menu", HEIGHT - 6, 6)
    
    def draw_demo(self):
        """Dessine le mode démo."""
        pyxel.cls(col=COL_BACKGROUND)
        
        if self.env:
            # Dessiner le serpent
            snake = self.env.get_snake_body()
            self.draw_snake(snake)
            
            # Dessiner la pomme
            apple = self.env.get_apple_position()
            pyxel.pset(apple.x, apple.y, col=COL_APPLE)
            
            # Score
            score = self.env.get_score()
            self.draw_score_bar(score)
        
        # Message
        if self.demo_done:
            self.draw_centered_text("SPACE: Rejouer", HEIGHT - 8, COL_TEXT)
        
        pyxel.text(1, HEIGHT - 8, "ESC: Menu", 6)
    
    def draw_centered_text(self, text, y, color):
        """Dessine du texte centré."""
        text_width = len(text) * pyxel.FONT_WIDTH
        x = (WIDTH - text_width) // 2
        pyxel.text(x, y, text, color)


def define_sound_and_music():
    """Définit les sons et la musique."""
    pyxel.sounds[0].set(
        notes="c3e3g3c4c4", tones="s", volumes="4", effects=("n" * 4 + "f"), speed=7
    )
    pyxel.sounds[1].set(
        notes="f3 b2 f2 b1  f1 f1 f1 f1",
        tones="p",
        volumes=("4" * 4 + "4321"),
        effects=("n" * 7 + "f"),
        speed=9,
    )


if __name__ == "__main__":
    SnakeRL()
