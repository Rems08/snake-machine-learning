# Environnement Snake pour apprentissage par renforcement
# Adaptation du jeu Snake pour un agent RL

from collections import deque, namedtuple
from typing import Tuple, Optional
import random

Point = namedtuple("Point", ["x", "y"])

# Actions possibles
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Mappage des actions vers des vecteurs de direction
ACTION_TO_DIRECTION = {
    UP: Point(0, -1),
    DOWN: Point(0, 1),
    LEFT: Point(-1, 0),
    RIGHT: Point(1, 0)
}


class SnakeEnvironment:
    """
    Environnement Snake pour l'apprentissage par renforcement.
    
    Modélisation RL:
    - États: (danger_front, danger_left, danger_right, apple_up, apple_down, 
              apple_left, apple_right, direction_up, direction_down, 
              direction_left, direction_right)
    - Actions: UP (0), DOWN (1), LEFT (2), RIGHT (3)
    - Récompenses: +10 (pomme), -10 (mort), -0.1 (pas normal)
    """
    
    def __init__(self, width: int = 40, height: int = 40, seed: Optional[int] = None):
        """
        Initialise l'environnement.
        
        Args:
            width: Largeur de la grille
            height: Hauteur de la grille (sans la zone de score)
            seed: Seed pour la reproductibilité
        """
        self.width = width
        self.height = height
        self.height_score = 10  # Espace réservé pour le score en haut
        
        if seed is not None:
            random.seed(seed)
        
        self.reset()
    
    def reset(self) -> Tuple:
        """
        Réinitialise l'environnement pour un nouvel épisode.
        
        Returns:
            L'état initial
        """
        # Position de départ au centre
        start_x = self.width // 2
        start_y = self.height // 2
        self.snake = deque([Point(start_x, start_y)])
        self.direction = ACTION_TO_DIRECTION[RIGHT]
        self.score = 0
        self.steps = 0
        self.generate_apple()
        
        return self.get_state()
    
    def generate_apple(self):
        """Génère une pomme à une position aléatoire libre."""
        snake_pixels = set(self.snake)
        
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.apple = Point(x, y)
            if self.apple not in snake_pixels:
                break
    
    def get_state(self) -> Tuple:
        """
        Génère l'état actuel pour l'agent.
        
        L'état est composé de:
        - 3 valeurs de danger (devant, gauche, droite)
        - 4 valeurs de position de la pomme (up, down, left, right)
        - 4 valeurs de direction actuelle (one-hot encoding)
        
        Returns:
            Tuple représentant l'état (11 valeurs booléennes/binaires)
        """
        head = self.snake[0]
        
        # Déterminer les directions relatives (devant, gauche, droite)
        point_front = Point(head.x + self.direction.x, head.y + self.direction.y)
        
        # Rotation de 90° à gauche
        point_left = Point(head.x - self.direction.y, head.y + self.direction.x)
        
        # Rotation de 90° à droite
        point_right = Point(head.x + self.direction.y, head.y - self.direction.x)
        
        # Vérifier les dangers
        danger_front = self._is_collision(point_front)
        danger_left = self._is_collision(point_left)
        danger_right = self._is_collision(point_right)
        
        # Position relative de la pomme
        apple_up = self.apple.y < head.y
        apple_down = self.apple.y > head.y
        apple_left = self.apple.x < head.x
        apple_right = self.apple.x > head.x
        
        # Direction actuelle (one-hot)
        dir_up = (self.direction == ACTION_TO_DIRECTION[UP])
        dir_down = (self.direction == ACTION_TO_DIRECTION[DOWN])
        dir_left = (self.direction == ACTION_TO_DIRECTION[LEFT])
        dir_right = (self.direction == ACTION_TO_DIRECTION[RIGHT])
        
        state = (
            int(danger_front),
            int(danger_left),
            int(danger_right),
            int(apple_up),
            int(apple_down),
            int(apple_left),
            int(apple_right),
            int(dir_up),
            int(dir_down),
            int(dir_left),
            int(dir_right)
        )
        
        return state
    
    def _is_collision(self, point: Point) -> bool:
        """
        Vérifie si un point est en collision (mur ou corps du serpent).
        
        Args:
            point: Le point à vérifier
            
        Returns:
            True si collision, False sinon
        """
        # Collision avec les murs
        if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
            return True
        
        # Collision avec le corps (sauf la queue qui va disparaître)
        if point in list(self.snake)[:-1]:
            return True
        
        return False
    
    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Effectue une action dans l'environnement.
        
        Args:
            action: L'action à effectuer (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            (nouvel_état, récompense, done)
        """
        self.steps += 1
        
        # Mise à jour de la direction (empêcher le demi-tour)
        new_direction = ACTION_TO_DIRECTION[action]
        opposite = Point(-self.direction.x, -self.direction.y)
        
        if new_direction != opposite:
            self.direction = new_direction
        
        # Déplacer le serpent
        old_head = self.snake[0]
        new_head = Point(old_head.x + self.direction.x, old_head.y + self.direction.y)
        
        # Vérifier la collision
        if self._is_collision(new_head):
            reward = -10  # Pénalité pour la mort
            done = True
            return self.get_state(), reward, done
        
        # Ajouter la nouvelle tête
        self.snake.appendleft(new_head)
        
        # Vérifier si on mange une pomme
        if new_head == self.apple:
            self.score += 1
            reward = 10  # Récompense pour manger une pomme
            self.generate_apple()
        else:
            # Retirer la queue si on ne mange pas
            self.snake.pop()
            reward = -0.1  # Petite pénalité pour encourager l'efficacité
        
        done = False
        new_state = self.get_state()
        
        return new_state, reward, done
    
    def get_action_space(self) -> int:
        """Retourne le nombre d'actions possibles."""
        return 4
    
    def get_snake_head(self) -> Point:
        """Retourne la position de la tête du serpent."""
        return self.snake[0]
    
    def get_snake_body(self) -> deque:
        """Retourne le corps complet du serpent."""
        return self.snake
    
    def get_apple_position(self) -> Point:
        """Retourne la position de la pomme."""
        return self.apple
    
    def get_score(self) -> int:
        """Retourne le score actuel."""
        return self.score
    
    def get_steps(self) -> int:
        """Retourne le nombre de pas effectués."""
        return self.steps
