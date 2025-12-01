"""
Environnement Snake pour l'apprentissage par renforcement.

Modélisation RL :
- États : (direction_serpent, danger_devant, danger_gauche, danger_droite, 
           pomme_devant, pomme_gauche, pomme_droite, pomme_derriere)
- Actions : 0=up, 1=down, 2=left, 3=right
- Récompenses : +10 manger pomme, -10 mourir, -0.1 par pas
"""

import random
from typing import Tuple, List, Optional
from enum import IntEnum


class Direction(IntEnum):
    """Directions possibles du serpent."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeEnvironment:
    """Environnement du jeu Snake."""
    
    def __init__(self, grid_size: int = 10, max_steps: int = 200, seed: Optional[int] = None):
        """
        Initialise l'environnement Snake.
        
        Args:
            grid_size: Taille de la grille (grid_size x grid_size)
            max_steps: Nombre maximum de pas par épisode
            seed: Graine pour la génération aléatoire
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        
        # État du jeu
        self.snake: List[Tuple[int, int]] = []
        self.apple: Optional[Tuple[int, int]] = None
        self.direction: Direction = Direction.RIGHT
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False
        
        # Métriques
        self.apples_eaten: int = 0
        
    def reset(self) -> Tuple:
        """
        Réinitialise l'environnement pour un nouvel épisode.
        
        Returns:
            État initial
        """
        # Initialiser le serpent au centre de la grille
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center - 1), (center, center - 2)]
        self.direction = Direction.RIGHT
        
        # Placer une pomme
        self._spawn_apple()
        
        # Réinitialiser les métriques
        self.score = 0
        self.steps = 0
        self.done = False
        self.apples_eaten = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, dict]:
        """
        Effectue une action dans l'environnement.
        
        Args:
            action: Action à effectuer (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            (nouvel_état, récompense, terminé, infos)
        """
        if self.done:
            raise ValueError("Episode terminé. Appelez reset() d'abord.")
        
        self.steps += 1
        
        # Mettre à jour la direction (empêcher le demi-tour)
        new_direction = Direction(action)
        if not self._is_opposite_direction(new_direction):
            self.direction = new_direction
        
        # Calculer la nouvelle position de la tête
        head_x, head_y = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)
        
        # Vérifier les collisions
        reward = -0.1  # Pénalité par défaut pour encourager l'efficacité
        
        # Collision avec le mur
        if not self._is_valid_position(new_head):
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done, self._get_info()
        
        # Collision avec le corps
        if new_head in self.snake[:-1]:  # Exclure la queue qui va bouger
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done, self._get_info()
        
        # Déplacer le serpent
        self.snake.insert(0, new_head)
        
        # Vérifier si le serpent mange la pomme
        if new_head == self.apple:
            reward = 10
            self.score += 10
            self.apples_eaten += 1
            self._spawn_apple()
        else:
            # Retirer la queue si pas de pomme mangée
            self.snake.pop()
        
        # Vérifier si le nombre maximum de pas est atteint
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._get_state(), reward, self.done, self._get_info()
    
    def _is_opposite_direction(self, new_direction: Direction) -> bool:
        """Vérifie si la nouvelle direction est opposée à la direction actuelle."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[self.direction] == new_direction
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Vérifie si une position est valide (dans la grille)."""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _spawn_apple(self):
        """Place une pomme sur une case libre."""
        free_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    free_positions.append((x, y))
        
        if free_positions:
            self.apple = self.rng.choice(free_positions)
        else:
            # Grille pleine (victoire totale)
            self.apple = None
            self.done = True
    
    def _get_state(self) -> Tuple:
        """
        Retourne l'état actuel sous forme de tuple.
        
        État : (direction, danger_devant, danger_gauche, danger_droite,
                pomme_devant, pomme_gauche, pomme_droite, pomme_derriere)
        """
        head_x, head_y = self.snake[0]
        
        # Vérifier les dangers dans chaque direction relative
        danger_straight = self._is_danger_in_direction(self.direction)
        danger_left = self._is_danger_in_direction(self._turn_left(self.direction))
        danger_right = self._is_danger_in_direction(self._turn_right(self.direction))
        
        # Position relative de la pomme
        if self.apple:
            apple_x, apple_y = self.apple
            
            # Calculer la direction de la pomme par rapport à la direction actuelle
            apple_front = False
            apple_left = False
            apple_right = False
            apple_back = False
            
            if self.direction == Direction.UP:
                apple_front = apple_y < head_y
                apple_back = apple_y > head_y
                apple_left = apple_x < head_x
                apple_right = apple_x > head_x
            elif self.direction == Direction.DOWN:
                apple_front = apple_y > head_y
                apple_back = apple_y < head_y
                apple_left = apple_x > head_x
                apple_right = apple_x < head_x
            elif self.direction == Direction.LEFT:
                apple_front = apple_x < head_x
                apple_back = apple_x > head_x
                apple_left = apple_y > head_y
                apple_right = apple_y < head_y
            else:  # RIGHT
                apple_front = apple_x > head_x
                apple_back = apple_x < head_x
                apple_left = apple_y < head_y
                apple_right = apple_y > head_y
        else:
            apple_front = apple_left = apple_right = apple_back = False
        
        return (
            int(self.direction),
            int(danger_straight),
            int(danger_left),
            int(danger_right),
            int(apple_front),
            int(apple_left),
            int(apple_right),
            int(apple_back)
        )
    
    def _is_danger_in_direction(self, direction: Direction) -> bool:
        """Vérifie s'il y a un danger (mur ou corps) dans une direction donnée."""
        head_x, head_y = self.snake[0]
        
        if direction == Direction.UP:
            next_pos = (head_x, head_y - 1)
        elif direction == Direction.DOWN:
            next_pos = (head_x, head_y + 1)
        elif direction == Direction.LEFT:
            next_pos = (head_x - 1, head_y)
        else:  # RIGHT
            next_pos = (head_x + 1, head_y)
        
        # Danger si hors de la grille ou collision avec le corps
        return not self._is_valid_position(next_pos) or next_pos in self.snake[:-1]
    
    def _turn_left(self, direction: Direction) -> Direction:
        """Retourne la direction à gauche de la direction actuelle."""
        turns = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP
        }
        return turns[direction]
    
    def _turn_right(self, direction: Direction) -> Direction:
        """Retourne la direction à droite de la direction actuelle."""
        turns = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP
        }
        return turns[direction]
    
    def _get_info(self) -> dict:
        """Retourne des informations supplémentaires sur l'état actuel."""
        return {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake),
            'apples_eaten': self.apples_eaten
        }
    
    def render(self) -> str:
        """
        Retourne une représentation textuelle de la grille.
        
        Returns:
            Chaîne de caractères représentant la grille
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Placer le corps du serpent
        for x, y in self.snake[1:]:
            grid[y][x] = 'o'
        
        # Placer la tête du serpent
        head_x, head_y = self.snake[0]
        grid[head_y][head_x] = 'H'
        
        # Placer la pomme
        if self.apple:
            apple_x, apple_y = self.apple
            grid[apple_y][apple_x] = 'A'
        
        # Convertir en chaîne
        result = '\n'.join([''.join(row) for row in grid])
        result += f'\nScore: {self.score} | Steps: {self.steps} | Length: {len(self.snake)}'
        return result
    
    def get_grid_state(self) -> List[List[int]]:
        """
        Retourne l'état de la grille sous forme de matrice pour la visualisation.
        
        Returns:
            Matrice où 0=vide, 1=corps, 2=tête, 3=pomme
        """
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Corps du serpent
        for x, y in self.snake[1:]:
            grid[y][x] = 1
        
        # Tête du serpent
        head_x, head_y = self.snake[0]
        grid[head_y][head_x] = 2
        
        # Pomme
        if self.apple:
            apple_x, apple_y = self.apple
            grid[apple_y][apple_x] = 3
        
        return grid
