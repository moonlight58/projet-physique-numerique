"""
test_forest.py - Tests unitaires pour le module forest
"""

import pytest
import numpy as np
from forest import ForestGrid, CellState

class TestForestGrid:
    """Tests pour la classe ForestGrid"""
    
    def test_initialization(self):
        """Test de l'initialisation de la grille"""
        forest = ForestGrid(size=50, tree_density=0.6, seed=42)
        
        assert forest.size == 50
        assert forest.tree_density == 0.6
        assert forest.timestep == 0
        assert forest.grid.shape == (50, 50)
        
        # Vérifier que la densité est approximativement correcte
        tree_count = np.sum(forest.grid == CellState.TREE.value)
        density = tree_count / (50 * 50)
        assert abs(density - 0.6) < 0.1  # Tolérance de 10%
    
    def test_initialization_seed(self):
        """Test que le seed donne des résultats reproductibles"""
        forest1 = ForestGrid(size=30, tree_density=0.5, seed=123)
        forest2 = ForestGrid(size=30, tree_density=0.5, seed=123)
        
        assert np.array_equal(forest1.grid, forest2.grid)
    
    def test_ignite_center(self):
        """Test de l'allumage au centre"""
        forest = ForestGrid(size=51, tree_density=1.0, seed=42)
        result = forest.ignite_center()
        
        center = 51 // 2
        assert forest.grid[center, center] == CellState.FIRE.value
        assert result == True
    
    def test_ignite_specific_position(self):
        """Test de l'allumage à une position spécifique"""
        forest = ForestGrid(size=30, tree_density=1.0, seed=42)
        result = forest.ignite(10, 15)
        
        assert forest.grid[10, 15] == CellState.FIRE.value
        assert result == True
    
    def test_ignite_empty_cell(self):
        """Test que l'allumage échoue sur une cellule vide"""
        forest = ForestGrid(size=30, tree_density=0.0, seed=42)
        result = forest.ignite(10, 10)
        
        assert result == False
        assert forest.grid[10, 10] == CellState.EMPTY.value
    
    def test_ignite_out_of_bounds(self):
        """Test de l'allumage hors limites"""
        forest = ForestGrid(size=30, tree_density=1.0, seed=42)
        result = forest.ignite(50, 50)
        
        assert result == False
    
    def test_ignite_random(self):
        """Test de l'allumage aléatoire"""
        forest = ForestGrid(size=30, tree_density=0.8, seed=42)
        count = forest.ignite_random(n=5)
        
        assert count <= 5
        assert np.sum(forest.grid == CellState.FIRE.value) == count
    
    def test_get_neighbors_von_neumann(self):
        """Test de la récupération des voisins (Von Neumann)"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        
        # Coin
        neighbors = forest.get_neighbors(0, 0, 'von_neumann')
        assert len(neighbors) == 2
        assert (1, 0) in neighbors
        assert (0, 1) in neighbors
        
        # Centre
        neighbors = forest.get_neighbors(5, 5, 'von_neumann')
        assert len(neighbors) == 4
        assert (4, 5) in neighbors
        assert (6, 5) in neighbors
        assert (5, 4) in neighbors
        assert (5, 6) in neighbors
    
    def test_get_neighbors_moore(self):
        """Test de la récupération des voisins (Moore)"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        
        # Centre
        neighbors = forest.get_neighbors(5, 5, 'moore')
        assert len(neighbors) == 8
        
        # Coin
        neighbors = forest.get_neighbors(0, 0, 'moore')
        assert len(neighbors) == 3
    
    def test_propagate_deterministic(self):
        """Test de propagation avec p=1.0 (déterministe)"""
        forest = ForestGrid(size=10, tree_density=1.0, seed=42)
        forest.ignite(5, 5)
        
        # Première propagation
        continues = forest.propagate(p=1.0, neighborhood='von_neumann')
        
        assert continues == True
        assert forest.timestep == 1
        assert forest.grid[5, 5] == CellState.ASH.value
        
        # Vérifier que les 4 voisins ont pris feu
        assert forest.grid[4, 5] == CellState.FIRE.value
        assert forest.grid[6, 5] == CellState.FIRE.value
        assert forest.grid[5, 4] == CellState.FIRE.value
        assert forest.grid[5, 6] == CellState.FIRE.value
    
    def test_propagate_no_fire(self):
        """Test de propagation avec p=0.0 (pas de propagation)"""
        forest = ForestGrid(size=10, tree_density=1.0, seed=42)
        forest.ignite(5, 5)
        
        continues = forest.propagate(p=0.0, neighborhood='von_neumann')
        
        assert continues == False
        assert forest.grid[5, 5] == CellState.ASH.value
        
        # Vérifier qu'aucun voisin n'a pris feu
        assert forest.grid[4, 5] == CellState.TREE.value
        assert forest.grid[6, 5] == CellState.TREE.value
    
    def test_simulate(self):
        """Test de simulation complète"""
        forest = ForestGrid(size=30, tree_density=0.8, seed=42)
        forest.ignite_center()
        
        steps = forest.simulate(p=0.6, neighborhood='von_neumann', max_steps=100)
        
        assert steps >= 0
        assert steps <= 100
        assert forest.timestep > 0
        assert np.sum(forest.grid == CellState.FIRE.value) == 0  # Feu éteint
    
    def test_get_statistics(self):
        """Test des statistiques"""
        forest = ForestGrid(size=20, tree_density=0.5, seed=42)
        initial_trees = np.sum(forest.grid == CellState.TREE.value)
        
        forest.ignite_center()
        forest.simulate(p=0.8, neighborhood='von_neumann')
        
        stats = forest.get_statistics()
        
        assert stats['total_cells'] == 400
        assert stats['initial_trees'] == initial_trees
        assert stats['burned_cells'] >= 0
        assert stats['burned_percentage'] >= 0
        assert stats['burned_percentage'] <= 100
        assert stats['duration'] == forest.timestep
    
    def test_history_recording(self):
        """Test de l'enregistrement de l'historique"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        forest.ignite_center()
        
        initial_length = len(forest.history['timesteps'])
        forest.propagate(p=0.5)
        
        assert len(forest.history['timesteps']) == initial_length + 1
        assert forest.history['timesteps'][-1] == forest.timestep
        
        # Vérifier que la somme des états est correcte
        total = (forest.history['empty'][-1] + 
                forest.history['trees'][-1] + 
                forest.history['fires'][-1] + 
                forest.history['ashes'][-1])
        assert total == forest.size * forest.size
    
    def test_reset(self):
        """Test de la réinitialisation"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        initial_grid = forest.grid.copy()
        
        forest.ignite_center()
        forest.simulate(p=0.5)
        
        forest.reset()
        
        assert forest.timestep == 0
        assert len(forest.history['timesteps']) == 1
        # Note: reset() crée une nouvelle grille, donc elle sera différente
        # sauf si on fixe le seed dans __init__
    
    def test_empty_forest(self):
        """Test avec une forêt vide"""
        forest = ForestGrid(size=20, tree_density=0.0, seed=42)
        result = forest.ignite_center()
        
        assert result == False
        assert np.sum(forest.grid == CellState.FIRE.value) == 0
    
    def test_full_forest(self):
        """Test avec une forêt pleine"""
        forest = ForestGrid(size=20, tree_density=1.0, seed=42)
        forest.ignite_center()
        steps = forest.simulate(p=1.0, neighborhood='von_neumann')
        
        stats = forest.get_statistics()
        # Avec p=1.0 et densité=1.0, tout devrait brûler
        assert stats['burned_percentage'] > 99.0
    
    def test_repr(self):
        """Test de la représentation string"""
        forest = ForestGrid(size=30, tree_density=0.6, seed=42)
        repr_str = repr(forest)
        
        assert "ForestGrid" in repr_str
        assert "size=30" in repr_str
        assert "density=0.60" in repr_str

class TestCellState:
    """Tests pour l'enum CellState"""
    
    def test_cell_states(self):
        """Test que tous les états sont définis"""
        assert CellState.EMPTY.value == 0
        assert CellState.TREE.value == 1
        assert CellState.FIRE.value == 2
        assert CellState.ASH.value == 3

# Tests d'intégration
class TestIntegration:
    """Tests d'intégration pour des scénarios complets"""
    
    def test_percolation_threshold(self):
        """Test que le seuil de percolation existe"""
        size = 50
        n_simulations = 10
        
        results_low_p = []
        results_high_p = []
        
        # Test avec p faible
        for _ in range(n_simulations):
            forest = ForestGrid(size=size, tree_density=0.6)
            forest.ignite_center()
            forest.simulate(p=0.2, neighborhood='von_neumann')
            stats = forest.get_statistics()
            results_low_p.append(stats['burned_percentage'])
        
        # Test avec p élevé
        for _ in range(n_simulations):
            forest = ForestGrid(size=size, tree_density=0.6)
            forest.ignite_center()
            forest.simulate(p=0.8, neighborhood='von_neumann')
            stats = forest.get_statistics()
            results_high_p.append(stats['burned_percentage'])
        
        # Vérifier qu'il y a une différence significative
        mean_low = np.mean(results_low_p)
        mean_high = np.mean(results_high_p)
        
        assert mean_high > mean_low
        assert mean_high - mean_low > 20  # Au moins 20% de différence
    
    def test_neighborhood_effect(self):
        """Test que le voisinage affecte la propagation"""
        size = 50
        n_simulations = 10
        
        results_von_neumann = []
        results_moore = []
        
        for _ in range(n_simulations):
            # Von Neumann
            forest = ForestGrid(size=size, tree_density=0.6, seed=42)
            forest.ignite_center()
            forest.simulate(p=0.5, neighborhood='von_neumann')
            stats = forest.get_statistics()
            results_von_neumann.append(stats['burned_percentage'])
            
            # Moore
            forest = ForestGrid(size=size, tree_density=0.6, seed=42)
            forest.ignite_center()
            forest.simulate(p=0.5, neighborhood='moore')
            stats = forest.get_statistics()
            results_moore.append(stats['burned_percentage'])
        
        # Moore devrait brûler davantage (plus de voisins)
        mean_vn = np.mean(results_von_neumann)
        mean_moore = np.mean(results_moore)
        
        assert mean_moore >= mean_vn

if __name__ == "__main__":
    # Pour exécuter les tests depuis la ligne de commande
    pytest.main([__file__, "-v"])
