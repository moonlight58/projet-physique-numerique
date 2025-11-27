"""
utils.py - Fonctions utilitaires pour le projet
"""

import numpy as np
import json
import pickle
from pathlib import Path


def bresenham_line(x0, y0, x1, y1):
    """
    Algorithme de Bresenham pour tracer une ligne.
    
    Args:
        x0, y0: Point de départ
        x1, y1: Point d'arrivée
    
    Returns:
        list: Liste de tuples (x, y) formant la ligne
    
    Examples:
        >>> points = bresenham_line(0, 0, 5, 3)
        >>> len(points)
        6
    """
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        points.append((x, y))
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x += sx
        
        if e2 < dx:
            err += dx
            y += sy
    
    return points


def generate_perlin_noise(shape, scale=10, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Génère un bruit de Perlin 2D (simplifié).
    
    Utile pour créer des cartes d'altitude réalistes.
    
    Args:
        shape: Tuple (height, width)
        scale: Échelle du bruit
        octaves: Nombre d'octaves
        persistence: Persistance entre octaves
        lacunarity: Lacunarité (fréquence entre octaves)
        seed: Graine aléatoire
    
    Returns:
        numpy.ndarray: Carte de bruit normalisée [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    noise = np.zeros(shape)
    
    for octave in range(octaves):
        freq = lacunarity ** octave
        amp = persistence ** octave
        
        # Génération simple avec sinus
        x = np.linspace(0, scale * freq, width)
        y = np.linspace(0, scale * freq, height)
        X, Y = np.meshgrid(x, y)
        
        # Combinaison de sinus avec phases aléatoires
        phase_x = np.random.random() * 2 * np.pi
        phase_y = np.random.random() * 2 * np.pi
        
        octave_noise = np.sin(X + phase_x) * np.sin(Y + phase_y)
        noise += amp * octave_noise
    
    # Normaliser entre 0 et 1
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def calculate_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.
    
    Args:
        point1: Tuple (x1, y1)
        point2: Tuple (x2, y2)
    
    Returns:
        float: Distance euclidienne
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_neighbors_in_radius(center, radius, grid_size):
    """
    Retourne toutes les cellules dans un rayon donné.
    
    Args:
        center: Tuple (x, y) du centre
        radius: Rayon en cellules
        grid_size: Taille de la grille
    
    Returns:
        list: Liste de tuples (x, y) dans le rayon
    """
    x0, y0 = center
    neighbors = []
    
    for x in range(max(0, x0 - radius), min(grid_size, x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(grid_size, y0 + radius + 1)):
            dist = calculate_distance((x, y), center)
            if dist <= radius:
                neighbors.append((x, y))
    
    return neighbors


def save_simulation(forest, filepath):
    """
    Sauvegarde une simulation complète.
    
    Args:
        forest: Instance de ForestGrid
        filepath: Chemin du fichier de sauvegarde
    """
    data = {
        'size': forest.size,
        'tree_density': forest.tree_density,
        'timestep': forest.timestep,
        'grid': forest.grid.tolist(),
        'history': {
            'timesteps': forest.history['timesteps'],
            'trees': forest.history['trees'],
            'fires': forest.history['fires'],
            'ashes': forest.history['ashes'],
            'empty': forest.history['empty']
        }
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Format de fichier non supporté: {filepath.suffix}")


def load_simulation(filepath):
    """
    Charge une simulation sauvegardée.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        dict: Données de simulation
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Format de fichier non supporté: {filepath.suffix}")
    
    # Convertir les listes en arrays
    data['grid'] = np.array(data['grid'])
    
    return data


def create_circular_mask(center, radius, shape):
    """
    Crée un masque circulaire.
    
    Args:
        center: Tuple (x, y) du centre
        radius: Rayon du cercle
        shape: Tuple (height, width) de la grille
    
    Returns:
        numpy.ndarray: Masque booléen
    """
    x0, y0 = center
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    return dist <= radius


def compute_fire_shape_metrics(forest):
    """
    Calcule des métriques sur la forme du feu.
    
    Args:
        forest: Instance de ForestGrid
    
    Returns:
        dict: Métriques (circularité, périmètre, aire, etc.)
    """
    from scipy.ndimage import label, find_objects
    
    # Identifier les zones brûlées (feu actif + cendres)
    burned = ((forest.grid == 2) | (forest.grid == 3)).astype(int)
    
    if burned.sum() == 0:
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'compactness': 0,
            'centroid': (0, 0)
        }
    
    # Trouver le plus grand cluster
    labeled, num_features = label(burned)
    
    if num_features == 0:
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'compactness': 0,
            'centroid': (0, 0)
        }
    
    # Prendre le plus grand cluster
    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    largest = np.argmax(sizes) + 1
    main_cluster = (labeled == largest)
    
    # Aire
    area = main_cluster.sum()
    
    # Périmètre (approximatif)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(main_cluster)
    perimeter = np.sum(main_cluster) - np.sum(eroded)
    
    # Circularité: 4π × aire / périmètre²
    # Vaut 1 pour un cercle parfait, < 1 sinon
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Compacité: périmètre / √aire
    compactness = perimeter / np.sqrt(area) if area > 0 else 0
    
    # Centroïde
    y_coords, x_coords = np.where(main_cluster)
    centroid = (np.mean(y_coords), np.mean(x_coords))
    
    return {
        'area': int(area),
        'perimeter': int(perimeter),
        'circularity': float(circularity),
        'compactness': float(compactness),
        'centroid': centroid
    }


def parallel_simulations(params_list, n_jobs=-1):
    """
    Exécute plusieurs simulations en parallèle.
    
    Args:
        params_list: Liste de dictionnaires de paramètres
        n_jobs: Nombre de processus (-1 = tous les CPU)
    
    Returns:
        list: Résultats des simulations
    """
    from multiprocessing import Pool, cpu_count
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    def run_single_simulation(params):
        from forest import ForestGrid
        
        forest = ForestGrid(
            size=params.get('size', 100),
            tree_density=params.get('tree_density', 0.6),
            seed=params.get('seed', None)
        )
        
        forest.ignite_center()
        forest.simulate(
            p=params.get('p', 0.5),
            neighborhood=params.get('neighborhood', 'von_neumann')
        )
        
        return forest.get_statistics()
    
    with Pool(n_jobs) as pool:
        results = pool.map(run_single_simulation, params_list)
    
    return results


def export_animation_frames(forest, output_dir, prefix='frame'):
    """
    Exporte chaque étape de l'historique comme image.
    
    Args:
        forest: Instance de ForestGrid avec historique
        output_dir: Dossier de sortie
        prefix: Préfixe des noms de fichiers
    """
    from visualization import plot_grid
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Restaurer chaque état depuis l'historique
    for i in range(len(forest.history['timesteps'])):
        # Reconstruire la grille à ce timestep
        # (simplifié - en pratique il faudrait sauvegarder chaque grille)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_grid(forest, ax=ax)
        
        filepath = output_dir / f'{prefix}_{i:04d}.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f'Saved: {filepath}')


class ProgressTracker:
    """Suivi de progression pour les simulations longues."""
    
    def __init__(self, total, desc='Progress'):
        self.total = total
        self.current = 0
        self.desc = desc
    
    def update(self, n=1):
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f'\r{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)', end='')
        
        if self.current >= self.total:
            print()  # Nouvelle ligne à la fin
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()  # Assurer une nouvelle ligne


def generate_test_scenarios():
    """
    Génère des scénarios de test standards.
    
    Returns:
        list: Liste de dictionnaires de paramètres
    """
    scenarios = []
    
    # Scénario 1: Densité faible
    scenarios.append({
        'name': 'Low density',
        'size': 100,
        'tree_density': 0.3,
        'p': 0.5,
        'expected_burned_pct': (0, 30)
    })
    
    # Scénario 2: Densité moyenne
    scenarios.append({
        'name': 'Medium density',
        'size': 100,
        'tree_density': 0.6,
        'p': 0.5,
        'expected_burned_pct': (30, 70)
    })
    
    # Scénario 3: Densité élevée
    scenarios.append({
        'name': 'High density',
        'size': 100,
        'tree_density': 0.9,
        'p': 0.5,
        'expected_burned_pct': (60, 100)
    })
    
    # Scénario 4: Probabilité faible
    scenarios.append({
        'name': 'Low probability',
        'size': 100,
        'tree_density': 0.6,
        'p': 0.2,
        'expected_burned_pct': (0, 30)
    })
    
    # Scénario 5: Probabilité élevée
    scenarios.append({
        'name': 'High probability',
        'size': 100,
        'tree_density': 0.6,
        'p': 0.9,
        'expected_burned_pct': (60, 100)
    })
    
    return scenarios
