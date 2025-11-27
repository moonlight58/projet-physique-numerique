"""
test_forest.py - Tests unitaires complets pour le module forest
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.forest import ForestGrid, CellState


class TestCellState:
    """Tests pour l'enum CellState"""
    
    def test_cell_states_values(self):
        """Test que tous les états ont les bonnes valeurs"""
        assert CellState.EMPTY.value == 0
        assert CellState.TREE.value == 1
        assert CellState.FIRE.value == 2
        assert CellState.ASH.value == 3
    
    def test_cell_states_unique(self):
        """Test que toutes les valeurs sont uniques"""
        values = [state.value for state in CellState]
        assert len(values) == len(set(values))


class TestForestGridInitialization:
    """Tests pour l'initialisation de ForestGrid"""
    
    def test_basic_initialization(self):
        """Test de l'initialisation de base"""
        forest = ForestGrid(size=50, tree_density=0.6, seed=42)
        
        assert forest.size == 50
        assert forest.tree_density == 0.6
        assert forest.timestep == 0
        assert forest.grid.shape == (50, 50)
        assert isinstance(forest.history, dict)
    
    def test_density_accuracy(self):
        """Test que la densité générée est proche de celle demandée"""
        for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
            forest = ForestGrid(size=100, tree_density=density, seed=42)
            actual_density = np.sum(forest.grid == CellState.TREE.value) / (100 * 100)
            
            # Tolérance de 15% pour les densités extrêmes
            tolerance = 0.15 if density < 0.3 or density > 0.7 else 0.1
            assert abs(actual_density - density) < tolerance
    
    def test_seed_reproducibility(self):
        """Test que le seed donne des résultats reproductibles"""
        forest1 = ForestGrid(size=30, tree_density=0.5, seed=123)
        forest2 = ForestGrid(size=30, tree_density=0.5, seed=123)
        forest3 = ForestGrid(size=30, tree_density=0.5, seed=456)
        
        assert np.array_equal(forest1.grid, forest2.grid)
        assert not np.array_equal(forest1.grid, forest3.grid)
    
    def test_invalid_size(self):
        """Test que les tailles invalides lèvent une erreur"""
        with pytest.raises(ValueError):
            ForestGrid(size=0, tree_density=0.5)
        
        with pytest.raises(ValueError):
            ForestGrid(size=-10, tree_density=0.5)
    
    def test_invalid_density(self):
        """Test que les densités invalides lèvent une erreur"""
        with pytest.raises(ValueError):
            ForestGrid(size=50, tree_density=-0.1)
        
        with pytest.raises(ValueError):
            ForestGrid(size=50, tree_density=1.5)
    
    def test_extreme_densities(self):
        """Test les densités extrêmes"""
        # Densité nulle
        forest_empty = ForestGrid(size=30, tree_density=0.0, seed=42)
        assert np.all(forest_empty.grid == CellState.EMPTY.value)
        
        # Densité maximale
        forest_full = ForestGrid(size=30, tree_density=1.0, seed=42)
        assert np.all(forest_full.grid == CellState.TREE.value)
    
    def test_history_initialization(self):
        """Test que l'historique est correctement initialisé"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        
        assert len(forest.history['timesteps']) == 1
        assert forest.history['timesteps'][0] == 0
        
        total = (forest.history['empty'][0] + 
                forest.history['trees'][0] + 
                forest.history['fires'][0] + 
                forest.history['ashes'][0])
        
        assert total == 20 * 20


class TestForestGridIgnition:
    """Tests pour l'allumage du feu"""
    
    def test_ignite_specific_position(self):
        """Test de l'allumage à une position spécifique"""
        forest = ForestGrid(size=30, tree_density=1.0, seed=42)
        result = forest.ignite(10, 15)
        
        assert result == True
        assert forest.grid[10, 15] == CellState.FIRE.value
    
    def test_ignite_empty_cell(self):
        """Test que l'allumage échoue sur une cellule vide"""
        forest = ForestGrid(size=30, tree_density=0.0, seed=42)
        result = forest.ignite(10, 10)
        
        assert result == False
        assert forest.grid[10, 10] == CellState.EMPTY.value
    
    def test_ignite_out_of_bounds(self):
        """Test de l'allumage hors limites"""
        forest = ForestGrid(size=30, tree_density=1.0, seed=42)
        
        assert forest.ignite(-1, 10) == False
        assert forest.ignite(10, -1) == False
        assert forest.ignite(50, 10) == False
        assert forest.ignite(10, 50) == False
    
    def test_ignite_center(self):
        """Test de l'allumage au centre"""
        forest = ForestGrid(size=51, tree_density=1.0, seed=42)
        result = forest.ignite_center()
        
        center = 51 // 2
        assert result == True
        assert forest.grid[center, center] == CellState.FIRE.value
    
    def test_ignite_center_different_sizes(self):
        """Test l'allumage au centre pour différentes tailles"""
        for size in [10, 50, 99, 100]:
            forest = ForestGrid(size=size, tree_density=1.0, seed=42)
            forest.ignite_center()
            
            center = size // 2
            assert forest.grid[center, center] == CellState.FIRE.value
    
    def test_ignite_random(self):
        """Test de l'allumage aléatoire"""
        forest = ForestGrid(size=30, tree_density=0.8, seed=42)
        count = forest.ignite_random(n=5)
        
        assert 0 <= count <= 5
        assert np.sum(forest.grid == CellState.FIRE.value) == count
    
    def test_ignite_random_more_than_trees(self):
        """Test qu'on ne peut pas enflammer plus d'arbres qu'il n'y en a"""
        forest = ForestGrid(size=10, tree_density=0.1, seed=42)
        total_trees = np.sum(forest.grid == CellState.TREE.value)
        
        count = forest.ignite_random(n=1000)
        assert count <= total_trees
    
    def test_ignite_random_no_trees(self):
        """Test l'allumage aléatoire sans arbres"""
        forest = ForestGrid(size=20, tree_density=0.0, seed=42)
        count = forest.ignite_random(n=5)
        
        assert count == 0
    
    def test_ignite_already_burning(self):
        """Test qu'on ne peut pas enflammer une cellule déjà en feu"""
        forest = ForestGrid(size=20, tree_density=1.0, seed=42)
        
        assert forest.ignite(10, 10) == True
        assert forest.ignite(10, 10) == False  # Déjà en feu


class TestForestGridNeighbors:
    """Tests pour la récupération des voisins"""
    
    def test_von_neumann_center(self):
        """Test voisinage von Neumann au centre"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        neighbors = forest.get_neighbors(5, 5, 'von_neumann')
        
        assert len(neighbors) == 4
        assert (4, 5) in neighbors
        assert (6, 5) in neighbors
        assert (5, 4) in neighbors
        assert (5, 6) in neighbors
    
    def test_von_neumann_corner(self):
        """Test voisinage von Neumann dans un coin"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        neighbors = forest.get_neighbors(0, 0, 'von_neumann')
        
        assert len(neighbors) == 2
        assert (1, 0) in neighbors
        assert (0, 1) in neighbors
    
    def test_von_neumann_edge(self):
        """Test voisinage von Neumann sur un bord"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        neighbors = forest.get_neighbors(0, 5, 'von_neumann')
        
        assert len(neighbors) == 3
    
    def test_moore_center(self):
        """Test voisinage Moore au centre"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        neighbors = forest.get_neighbors(5, 5, 'moore')
        
        assert len(neighbors) == 8
        
        # Vérifier les 8 voisins
        expected = [
            (4, 5), (6, 5), (5, 4), (5, 6),  # Von Neumann
            (4, 4), (4, 6), (6, 4), (6, 6)   # Diagonales
        ]
        
        for pos in expected:
            assert pos in neighbors
    
    def test_moore_corner(self):
        """Test voisinage Moore dans un coin"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        neighbors = forest.get_neighbors(0, 0, 'moore')
        
        assert len(neighbors) == 3
        assert (1, 0) in neighbors
        assert (0, 1) in neighbors
        assert (1, 1) in neighbors
    
    def test_invalid_neighborhood(self):
        """Test qu'un voisinage invalide lève une erreur"""
        forest = ForestGrid(size=10, tree_density=0.5, seed=42)
        
        with pytest.raises(ValueError):
            forest.get_neighbors(5, 5, 'invalid')


class TestForestGridPropagation:
    """Tests pour la propagation du feu"""
    
    def test_propagate_deterministic(self):
        """Test propagation déterministe (p=1.0)"""
        forest = ForestGrid(size=10, tree_density=1.0, seed=42)
        forest.ignite(5, 5)
        
        continues = forest.propagate(p=1.0, neighborhood='von_neumann')
        
        assert continues == True
        assert forest.timestep == 1
        assert forest.grid[5, 5] == CellState.ASH.value
        
        # Les 4 voisins doivent avoir pris feu
        assert forest.grid[4, 5] == CellState.FIRE.value
        assert forest.grid[6, 5] == CellState.FIRE.value
        assert forest.grid[5, 4] == CellState.FIRE.value
        assert forest.grid[5, 6] == CellState.FIRE.value
    
    def test_propagate_no_propagation(self):
        """Test sans propagation (p=0.0)"""
        forest = ForestGrid(size=10, tree_density=1.0, seed=42)
        forest.ignite(5, 5)
        
        continues = forest.propagate(p=0.0, neighborhood='von_neumann')
        
        assert continues == False
        assert forest.grid[5, 5] == CellState.ASH.value
        
        # Aucun voisin ne doit avoir pris feu
        assert forest.grid[4, 5] == CellState.TREE.value
        assert forest.grid[6, 5] == CellState.TREE.value
    
    def test_propagate_moore_vs_von_neumann(self):
        """Test que Moore propage plus que von Neumann"""
        # Von Neumann
        forest_vn = ForestGrid(size=20, tree_density=1.0, seed=42)
        forest_vn.ignite_center()
        forest_vn.propagate(p=1.0, neighborhood='von_neumann')
        fires_vn = np.sum(forest_vn.grid == CellState.FIRE.value)
        
        # Moore
        forest_moore = ForestGrid(size=20, tree_density=1.0, seed=42)
        forest_moore.ignite_center()
        forest_moore.propagate(p=1.0, neighborhood='moore')
        fires_moore = np.sum(forest_moore.grid == CellState.FIRE.value)
        
        assert fires_moore > fires_vn
    
    def test_propagate_updates_history(self):
        """Test que la propagation met à jour l'historique"""
        forest = ForestGrid(size=20, tree_density=0.8, seed=42)
        forest.ignite_center()
        
        initial_length = len(forest.history['timesteps'])
        forest.propagate(p=0.5)
        
        assert len(forest.history['timesteps']) == initial_length + 1
        assert forest.history['timesteps'][-1] == forest.timestep
    
    def test_propagate_invalid_probability(self):
        """Test que les probabilités invalides lèvent une erreur"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        forest.ignite_center()
        
        with pytest.raises(ValueError):
            forest.propagate(p=-0.1)
        
        with pytest.raises(ValueError):
            forest.propagate(p=1.5)
    
    def test_propagate_vectorized_vs_loop(self):
        """Test que les deux méthodes donnent les mêmes résultats"""
        # Version vectorisée
        forest1 = ForestGrid(size=30, tree_density=0.6, seed=42)
        forest1.ignite_center()
        forest1.propagate(p=0.8, use_vectorized=True)
        
        # Version en boucle
        forest2 = ForestGrid(size=30, tree_density=0.6, seed=42)
        forest2.ignite_center()
        forest2.propagate(p=0.8, use_vectorized=False)
        
        # Les résultats doivent être identiques
        assert np.array_equal(forest1.grid, forest2.grid)


class TestForestGridSimulation:
    """Tests pour la simulation complète"""
    
    def test_simulate_basic(self):
        """Test simulation de base"""
        forest = ForestGrid(size=30, tree_density=0.8, seed=42)
        forest.ignite_center()
        
        steps = forest.simulate(p=0.6, max_steps=100)
        
        assert 0 <= steps <= 100
        assert forest.timestep > 0
        assert np.sum(forest.grid == CellState.FIRE.value) == 0  # Feu éteint
    
    def test_simulate_high_probability(self):
        """Test que p élevé brûle plus"""
        results_low = []
        results_high = []
        
        for i in range(10):
            # p faible
            forest = ForestGrid(size=50, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.2)
            results_low.append(forest.get_statistics()['burned_percentage'])
            
            # p élevé
            forest = ForestGrid(size=50, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.9)
            results_high.append(forest.get_statistics()['burned_percentage'])
        
        assert np.mean(results_high) > np.mean(results_low)
    
    def test_simulate_max_steps(self):
        """Test que max_steps est respecté"""
        forest = ForestGrid(size=100, tree_density=1.0, seed=42)
        forest.ignite_center()
        
        steps = forest.simulate(p=1.0, max_steps=10)
        
        assert steps <= 10
    
    def test_simulate_callback(self):
        """Test du callback pendant la simulation"""
        forest = ForestGrid(size=30, tree_density=0.8, seed=42)
        forest.ignite_center()
        
        callback_calls = []
        
        def my_callback(f, step):
            callback_calls.append(step)
            return step < 5  # Arrêter après 5 pas
        
        steps = forest.simulate(p=0.5, callback=my_callback)
        
        assert len(callback_calls) > 0
        assert steps <= 5


class TestForestGridStatistics:
    """Tests pour les statistiques"""
    
    def test_statistics_structure(self):
        """Test que les statistiques ont la bonne structure"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        stats = forest.get_statistics()
        
        required_keys = [
            'total_cells', 'initial_trees', 'burned_cells',
            'burned_percentage', 'burned_trees_percentage',
            'duration', 'final_trees', 'max_fire_intensity'
        ]
        
        for key in required_keys:
            assert key in stats
    
    def test_statistics_values(self):
        """Test que les valeurs statistiques sont cohérentes"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        forest.ignite_center()
        forest.simulate(p=0.8)
        
        stats = forest.get_statistics()
        
        assert stats['total_cells'] == 400
        assert 0 <= stats['burned_percentage'] <= 100
        assert 0 <= stats['burned_trees_percentage'] <= 100
        assert stats['duration'] == forest.timestep
        assert stats['burned_cells'] == forest.history['ashes'][-1]
    
    def test_statistics_full_burn(self):
        """Test statistiques pour combustion complète"""
        forest = ForestGrid(size=20, tree_density=1.0, seed=42)
        forest.ignite_center()
        forest.simulate(p=1.0)
        
        stats = forest.get_statistics()
        
        # Presque tout devrait brûler
        assert stats['burned_percentage'] > 95.0


class TestForestGridUtilities:
    """Tests pour les méthodes utilitaires"""
    
    def test_reset(self):
        """Test de la réinitialisation"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        initial_grid = forest.grid.copy()
        
        forest.ignite_center()
        forest.simulate(p=0.5)
        
        assert forest.timestep > 0
        
        forest.reset()
        
        assert forest.timestep == 0
        assert len(forest.history['timesteps']) == 1
    
    def test_copy(self):
        """Test de la copie"""
        forest = ForestGrid(size=20, tree_density=0.6, seed=42)
        forest.ignite_center()
        forest.propagate(p=0.5)
        
        forest_copy = forest.copy()
        
        # Vérifier que c'est une copie indépendante
        assert np.array_equal(forest.grid, forest_copy.grid)
        assert forest.timestep == forest_copy.timestep
        
        # Modifier la copie
        forest_copy.propagate(p=0.5)
        
        # L'original ne doit pas être modifié
        assert not np.array_equal(forest.grid, forest_copy.grid)
    
    def test_repr(self):
        """Test de la représentation string"""
        forest = ForestGrid(size=30, tree_density=0.6, seed=42)
        repr_str = repr(forest)
        
        assert "ForestGrid" in repr_str
        assert "size=30" in repr_str
        assert "density=0.60" in repr_str
    
    def test_str(self):
        """Test de la version lisible"""
        forest = ForestGrid(size=30, tree_density=0.6, seed=42)
        str_repr = str(forest)
        
        assert "30×30" in str_repr or "30x30" in str_repr


class TestIntegration:
    """Tests d'intégration pour des scénarios complets"""
    
    def test_percolation_threshold_exists(self):
        """Test qu'un seuil de percolation existe"""
        size = 50
        n_sim = 5
        
        results_low = []
        results_high = []
        
        for i in range(n_sim):
            # p faible
            forest = ForestGrid(size=size, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.2)
            results_low.append(forest.get_statistics()['burned_percentage'])
            
            # p élevé
            forest = ForestGrid(size=size, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.8)
            results_high.append(forest.get_statistics()['burned_percentage'])
        
        mean_low = np.mean(results_low)
        mean_high = np.mean(results_high)
        
        assert mean_high > mean_low
        assert mean_high - mean_low > 15  # Au moins 15% de différence
    
    def test_neighborhood_effect(self):
        """Test que le voisinage affecte significativement la propagation"""
        n_sim = 5
        results_vn = []
        results_moore = []
        
        for i in range(n_sim):
            # Von Neumann
            forest = ForestGrid(size=50, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.5, neighborhood='von_neumann')
            results_vn.append(forest.get_statistics()['burned_percentage'])
            
            # Moore
            forest = ForestGrid(size=50, tree_density=0.6, seed=i)
            forest.ignite_center()
            forest.simulate(p=0.5, neighborhood='moore')
            results_moore.append(forest.get_statistics()['burned_percentage'])
        
        # Moore devrait brûler plus (plus de voisins)
        assert np.mean(results_moore) >= np.mean(results_vn)
    
    def test_density_effect(self):
        """Test que la densité affecte la propagation"""
        results = {}
        
        for density in [0.3, 0.6, 0.9]:
            burned_list = []
            for i in range(5):
                forest = ForestGrid(size=50, tree_density=density, seed=i)
                forest.ignite_center()
                forest.simulate(p=0.5)
                burned_list.append(forest.get_statistics()['burned_percentage'])
            
            results[density] = np.mean(burned_list)
        
        # Plus de densité = plus de surface brûlée
        assert results[0.9] > results[0.6]
        assert results[0.6] > results[0.3]


# Tests de performance (optionnels)
@pytest.mark.slow
class TestPerformance:
    """Tests de performance (marqués comme lents)"""
    
    def test_vectorized_faster_than_loop(self):
        """Test que la version vectorisée est plus rapide"""
        import time
        
        # Version vectorisée
        forest = ForestGrid(size=100, tree_density=0.6, seed=42)
        forest.ignite_center()
        
        start = time.time()
        for _ in range(10):
            forest.propagate(p=0.5, use_vectorized=True)
        time_vectorized = time.time() - start
        
        # Version en boucle
        forest = ForestGrid(size=100, tree_density=0.6, seed=42)
        forest.ignite_center()
        
        start = time.time()
        for _ in range(10):
            forest.propagate(p=0.5, use_vectorized=False)
        time_loop = time.time() - start
        
        # La version vectorisée devrait être au moins 2x plus rapide
        assert time_vectorized < time_loop / 2


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v", "--tb=short"])
