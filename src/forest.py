"""
forest.py - Classe principale pour la grille de forêt
"""

import numpy as np
from enum import Enum

class CellState(Enum):
    """États possibles d'une cellule"""
    EMPTY = 0    # Vide (sol nu)
    TREE = 1     # Arbre (combustible)
    FIRE = 2     # En feu
    ASH = 3      # Cendres (brûlé)

class ForestGrid:
    """
    Représente une forêt sur une grille 2D avec simulation de feu.
    """
    
    def __init__(self, size, tree_density=0.6, seed=None):
        """
        Initialise la grille de forêt.
        
        Args:
            size (int): Taille de la grille (size × size)
            tree_density (float): Proportion d'arbres initiaux [0, 1]
            seed (int): Graine pour la reproductibilité
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.size = size
        self.tree_density = tree_density
        self.timestep = 0
        
        # Initialisation de la grille
        self.grid = np.random.choice(
            [CellState.EMPTY.value, CellState.TREE.value],
            size=(size, size),
            p=[1 - tree_density, tree_density]
        )
        
        # Historique pour les statistiques
        self.history = {
            'timesteps': [],
            'trees': [],
            'fires': [],
            'ashes': [],
            'empty': []
        }
        
        self._record_state()
    
    def ignite(self, x, y):
        """
        Déclenche un feu à la position (x, y).
        
        Args:
            x, y (int): Coordonnées de la cellule à enflammer
        
        Returns:
            bool: True si l'ignition a réussi, False sinon
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.grid[x, y] == CellState.TREE.value:
                self.grid[x, y] = CellState.FIRE.value
                return True
        return False
    
    def ignite_center(self):
        """Déclenche un feu au centre de la grille."""
        center = self.size // 2
        return self.ignite(center, center)
    
    def ignite_random(self, n=1):
        """
        Déclenche n feux à des positions aléatoires.
        
        Args:
            n (int): Nombre de feux à déclencher
        
        Returns:
            int: Nombre de feux effectivement déclenchés
        """
        count = 0
        tree_positions = np.argwhere(self.grid == CellState.TREE.value)
        
        if len(tree_positions) == 0:
            return 0
        
        n = min(n, len(tree_positions))
        selected = tree_positions[np.random.choice(len(tree_positions), n, replace=False)]
        
        for x, y in selected:
            if self.ignite(x, y):
                count += 1
        
        return count
    
    def get_neighbors(self, x, y, neighborhood='von_neumann'):
        """
        Retourne les coordonnées des voisins d'une cellule.
        
        Args:
            x, y (int): Coordonnées de la cellule
            neighborhood (str): 'von_neumann' (4 voisins) ou 'moore' (8 voisins)
        
        Returns:
            list: Liste de tuples (x, y) des voisins valides
        """
        if neighborhood == 'von_neumann':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif neighborhood == 'moore':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            raise ValueError(f"Neighborhood inconnu: {neighborhood}")
        
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def propagate(self, p=0.5, neighborhood='von_neumann'):
        """
        Propage le feu pendant un pas de temps (modèle de percolation simple).
        
        Args:
            p (float): Probabilité de propagation [0, 1]
            neighborhood (str): Type de voisinage
        
        Returns:
            bool: True si le feu continue, False si éteint
        """
        new_grid = self.grid.copy()
        fire_cells = np.argwhere(self.grid == CellState.FIRE.value)
        
        # Propagation depuis chaque cellule en feu
        for x, y in fire_cells:
            neighbors = self.get_neighbors(x, y, neighborhood)
            
            for nx, ny in neighbors:
                # Si le voisin est un arbre, il peut prendre feu
                if self.grid[nx, ny] == CellState.TREE.value:
                    if np.random.random() < p:
                        new_grid[nx, ny] = CellState.FIRE.value
            
            # Le feu s'éteint après avoir brûlé
            new_grid[x, y] = CellState.ASH.value
        
        self.grid = new_grid
        self.timestep += 1
        self._record_state()
        
        # Le feu continue s'il reste des cellules en feu
        return np.sum(self.grid == CellState.FIRE.value) > 0
    
    def simulate(self, p=0.5, neighborhood='von_neumann', max_steps=1000):
        """
        Simule la propagation complète du feu jusqu'à extinction.
        
        Args:
            p (float): Probabilité de propagation
            neighborhood (str): Type de voisinage
            max_steps (int): Nombre maximum de pas de temps
        
        Returns:
            int: Nombre de pas de temps effectués
        """
        steps = 0
        while self.propagate(p, neighborhood) and steps < max_steps:
            steps += 1
        
        return steps
    
    def _record_state(self):
        """Enregistre l'état actuel dans l'historique."""
        self.history['timesteps'].append(self.timestep)
        self.history['empty'].append(np.sum(self.grid == CellState.EMPTY.value))
        self.history['trees'].append(np.sum(self.grid == CellState.TREE.value))
        self.history['fires'].append(np.sum(self.grid == CellState.FIRE.value))
        self.history['ashes'].append(np.sum(self.grid == CellState.ASH.value))
    
    def get_statistics(self):
        """
        Retourne les statistiques de la simulation.
        
        Returns:
            dict: Dictionnaire avec les métriques clés
        """
        total_cells = self.size * self.size
        initial_trees = self.history['trees'][0]
        final_ashes = self.history['ashes'][-1]
        
        return {
            'total_cells': total_cells,
            'initial_trees': initial_trees,
            'burned_cells': final_ashes,
            'burned_percentage': (final_ashes / total_cells) * 100 if total_cells > 0 else 0,
            'burned_trees_percentage': (final_ashes / initial_trees) * 100 if initial_trees > 0 else 0,
            'duration': self.timestep,
            'final_trees': self.history['trees'][-1]
        }
    
    def reset(self):
        """Réinitialise la grille à son état initial."""
        self.__init__(self.size, self.tree_density)
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"ForestGrid(size={self.size}, density={self.tree_density:.2f}, "
                f"timestep={self.timestep}, burned={stats['burned_cells']})")
