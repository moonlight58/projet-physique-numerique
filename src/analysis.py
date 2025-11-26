"""
analysis.py - Fonctions d'analyse statistique et de percolation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from forest import ForestGrid

def find_percolation_threshold(size=100, p_range=None, n_simulations=50, 
                                neighborhood='von_neumann', threshold=0.5):
    """
    Trouve le seuil de percolation critique.
    
    Le seuil de percolation est la probabilité minimale pour laquelle
    le feu se propage significativement (> threshold de la forêt brûlée).
    
    Args:
        size (int): Taille de la grille
        p_range (array): Valeurs de probabilité à tester
        n_simulations (int): Nombre de simulations par probabilité
        neighborhood (str): Type de voisinage
        threshold (float): Seuil de propagation significative
    
    Returns:
        dict: Résultats avec probabilités et proportions brûlées
    """
    if p_range is None:
        p_range = np.linspace(0.0, 1.0, 21)
    
    mean_burned = []
    std_burned = []
    
    print(f"Searching for percolation threshold (size={size}, n_sim={n_simulations})...")
    
    for i, p in enumerate(p_range):
        burned_list = []
        
        for _ in range(n_simulations):
            forest = ForestGrid(size, tree_density=0.6)
            forest.ignite_center()
            forest.simulate(p=p, neighborhood=neighborhood)
            
            stats = forest.get_statistics()
            # Proportion par rapport aux arbres initiaux
            burned_pct = stats['burned_trees_percentage']
            burned_list.append(burned_pct)
        
        mean_burned.append(np.mean(burned_list))
        std_burned.append(np.std(burned_list))
        
        print(f"p={p:.2f}: {mean_burned[-1]:.1f}% ± {std_burned[-1]:.1f}%")
    
    mean_burned = np.array(mean_burned)
    std_burned = np.array(std_burned)
    
    # Estimation du seuil critique (où 50% des arbres brûlent)
    idx_critical = np.argmin(np.abs(mean_burned - threshold * 100))
    p_critical = p_range[idx_critical]
    
    print(f"\nEstimated critical probability p_c ≈ {p_critical:.3f}")
    
    return {
        'probabilities': p_range,
        'mean_burned': mean_burned,
        'std_burned': std_burned,
        'p_critical': p_critical,
        'threshold': threshold
    }

def plot_percolation_curve(results):
    """
    Affiche la courbe de percolation.
    
    Args:
        results (dict): Résultats de find_percolation_threshold()
    
    Returns:
        fig: Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    p = results['probabilities']
    mean = results['mean_burned']
    std = results['std_burned']
    p_c = results['p_critical']
    
    # Courbe principale avec barres d'erreur
    ax.errorbar(p, mean, yerr=std, fmt='o-', capsize=5, capthick=2,
               color='darkblue', ecolor='lightblue', linewidth=2,
               markersize=8, label='Mean ± Std')
    
    # Ligne du seuil critique
    ax.axvline(p_c, color='red', linestyle='--', linewidth=2,
              label=f'Critical threshold p_c ≈ {p_c:.3f}')
    ax.axhline(results['threshold'] * 100, color='green', linestyle='--',
              linewidth=2, alpha=0.5, label=f'{results["threshold"]*100}% threshold')
    
    ax.set_xlabel('Propagation probability (p)', fontsize=14)
    ax.set_ylabel('Burned trees (%)', fontsize=14)
    ax.set_title('Percolation curve: Burned area vs Propagation probability',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return fig

def analyze_density_effect(size=100, densities=None, p=0.5, 
                           n_simulations=30, neighborhood='von_neumann'):
    """
    Analyse l'effet de la densité d'arbres sur la propagation.
    
    Args:
        size (int): Taille de la grille
        densities (array): Densités à tester
        p (float): Probabilité de propagation
        n_simulations (int): Nombre de simulations par densité
        neighborhood (str): Type de voisinage
    
    Returns:
        dict: Résultats avec densités et statistiques
    """
    if densities is None:
        densities = np.linspace(0.1, 1.0, 10)
    
    mean_burned = []
    std_burned = []
    mean_duration = []
    
    print(f"Analyzing density effect (p={p})...")
    
    for i, d in enumerate(densities):
        burned_list = []
        duration_list = []
        
        for _ in range(n_simulations):
            forest = ForestGrid(size, tree_density=d)
            forest.ignite_center()
            forest.simulate(p=p, neighborhood=neighborhood)
            
            stats = forest.get_statistics()
            burned_list.append(stats['burned_percentage'])
            duration_list.append(stats['duration'])
        
        mean_burned.append(np.mean(burned_list))
        std_burned.append(np.std(burned_list))
        mean_duration.append(np.mean(duration_list))
        
        print(f"d={d:.2f}: {mean_burned[-1]:.1f}% ± {std_burned[-1]:.1f}% "
              f"(duration: {mean_duration[-1]:.1f} steps)")
    
    return {
        'densities': densities,
        'mean_burned': np.array(mean_burned),
        'std_burned': np.array(std_burned),
        'mean_duration': np.array(mean_duration)
    }

def plot_density_analysis(results):
    """
    Affiche les résultats de l'analyse de densité.
    
    Args:
        results (dict): Résultats de analyze_density_effect()
    
    Returns:
        fig: Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    d = results['densities']
    mean_b = results['mean_burned']
    std_b = results['std_burned']
    duration = results['mean_duration']
    
    # Graphique 1: Surface brûlée
    ax1.errorbar(d, mean_b, yerr=std_b, fmt='o-', capsize=5, capthick=2,
                color='darkred', ecolor='lightcoral', linewidth=2,
                markersize=8)
    ax1.set_xlabel('Tree density', fontsize=14)
    ax1.set_ylabel('Burned area (%)', fontsize=14)
    ax1.set_title('Burned area vs Tree density', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 105)
    
    # Graphique 2: Durée de propagation
    ax2.plot(d, duration, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax2.set_xlabel('Tree density', fontsize=14)
    ax2.set_ylabel('Fire duration (timesteps)', fontsize=14)
    ax2.set_title('Fire duration vs Tree density', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig

def cluster_analysis(forest):
    """
    Analyse les clusters (amas) de cellules brûlées.
    
    Utilise un algorithme de flood-fill pour identifier les clusters connectés.
    
    Args:
        forest (ForestGrid): ForestGrid après simulation
    
    Returns:
        dict: Statistiques sur les clusters
    """
    grid = forest.grid
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []
    
    def flood_fill(x, y, target_state):
        """Remplit récursivement un cluster."""
        stack = [(x, y)]
        cluster_size = 0
        
        while stack:
            cx, cy = stack.pop()
            
            if (cx < 0 or cx >= forest.size or 
                cy < 0 or cy >= forest.size or 
                visited[cx, cy] or 
                grid[cx, cy] != target_state):
                continue
            
            visited[cx, cy] = True
            cluster_size += 1
            
            # 4-voisinage
            stack.extend([(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)])
        
        return cluster_size
    
    # Trouver tous les clusters de cendres (zones brûlées)
    for i in range(forest.size):
        for j in range(forest.size):
            if grid[i, j] == 3 and not visited[i, j]:  # ASH
                size = flood_fill(i, j, 3)
                if size > 0:
                    clusters.append(size)
    
    if len(clusters) == 0:
        return {
            'n_clusters': 0,
            'sizes': [],
            'mean_size': 0,
            'max_size': 0,
            'total_burned': 0
        }
    
    clusters = np.array(clusters)
    
    return {
        'n_clusters': len(clusters),
        'sizes': clusters,
        'mean_size': np.mean(clusters),
        'max_size': np.max(clusters),
        'total_burned': np.sum(clusters)
    }

def plot_cluster_distribution(results_list, labels=None):
    """
    Affiche la distribution des tailles de clusters.
    
    Args:
        results_list (list): Liste de dict de cluster_analysis()
        labels (list): Labels pour chaque ensemble de résultats
    
    Returns:
        fig: Figure matplotlib
    """
    if not isinstance(results_list, list):
        results_list = [results_list]
    
    if labels is None:
        labels = [f'Simulation {i+1}' for i in range(len(results_list))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Histogrammes des tailles
    for results, label in zip(results_list, labels):
        if len(results['sizes']) > 0:
            ax1.hist(results['sizes'], bins=30, alpha=0.6, label=label, 
                    edgecolor='black')
    
    ax1.set_xlabel('Cluster size (cells)', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Distribution of cluster sizes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Log-log plot (loi de puissance?)
    for results, label in zip(results_list, labels):
        if len(results['sizes']) > 0:
            sizes = np.sort(results['sizes'])[::-1]
            ranks = np.arange(1, len(sizes) + 1)
            ax2.loglog(ranks, sizes, 'o-', alpha=0.7, label=label, markersize=4)
    
    ax2.set_xlabel('Rank', fontsize=14)
    ax2.set_ylabel('Cluster size (cells)', fontsize=14)
    ax2.set_title('Rank-size distribution (log-log)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig

def monte_carlo_statistics(size=100, tree_density=0.6, p=0.5, 
                           n_simulations=1000, neighborhood='von_neumann'):
    """
    Effectue une analyse Monte-Carlo complète.
    
    Args:
        size (int): Taille de la grille
        tree_density (float): Densité d'arbres
        p (float): Probabilité de propagation
        n_simulations (int): Nombre de simulations
        neighborhood (str): Type de voisinage
    
    Returns:
        dict: Statistiques complètes
    """
    burned_list = []
    duration_list = []
    
    print(f"Running Monte-Carlo simulation ({n_simulations} runs)...")
    
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{n_simulations}")
        
        forest = ForestGrid(size, tree_density)
        forest.ignite_center()
        forest.simulate(p=p, neighborhood=neighborhood)
        
        stats = forest.get_statistics()
        burned_list.append(stats['burned_percentage'])
        duration_list.append(stats['duration'])
    
    burned_array = np.array(burned_list)
    duration_array = np.array(duration_list)
    
    return {
        'n_simulations': n_simulations,
        'burned': {
            'data': burned_array,
            'mean': np.mean(burned_array),
            'std': np.std(burned_array),
            'median': np.median(burned_array),
            'min': np.min(burned_array),
            'max': np.max(burned_array),
            'q25': np.percentile(burned_array, 25),
            'q75': np.percentile(burned_array, 75)
        },
        'duration': {
            'data': duration_array,
            'mean': np.mean(duration_array),
            'std': np.std(duration_array),
            'median': np.median(duration_array),
            'min': np.min(duration_array),
            'max': np.max(duration_array)
        }
    }

def plot_monte_carlo(results):
    """
    Affiche les résultats Monte-Carlo.
    
    Args:
        results (dict): Résultats de monte_carlo_statistics()
    
    Returns:
        fig: Figure matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    burned_data = results['burned']['data']
    duration_data = results['duration']['data']
    
    # Histogramme: surface brûlée
    axes[0, 0].hist(burned_data, bins=50, color='darkred', 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(results['burned']['mean'], color='blue', 
                       linestyle='--', linewidth=2, label=f"Mean: {results['burned']['mean']:.2f}%")
    axes[0, 0].axvline(results['burned']['median'], color='green', 
                       linestyle='--', linewidth=2, label=f"Median: {results['burned']['median']:.2f}%")
    axes[0, 0].set_xlabel('Burned area (%)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of burned areas', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Histogramme: durée
    axes[0, 1].hist(duration_data, bins=50, color='darkorange', 
                    alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(results['duration']['mean'], color='blue', 
                       linestyle='--', linewidth=2, label=f"Mean: {results['duration']['mean']:.2f}")
    axes[0, 1].set_xlabel('Fire duration (timesteps)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of fire durations', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Boxplot
    bp = axes[1, 0].boxplot([burned_data, duration_data], 
                            labels=['Burned area (%)', 'Duration (steps)'],
                            patch_artist=True)
    bp['boxes'][0].set_facecolor('darkred')
    bp['boxes'][1].set_facecolor('darkorange')
    axes[1, 0].set_title('Statistical summary', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot: corrélation durée vs surface brûlée
    axes[1, 1].scatter(duration_data, burned_data, alpha=0.3, s=10)
    
    # Régression linéaire
    z = np.polyfit(duration_data, burned_data, 1)
    p = np.poly1d(z)
    x_line = np.linspace(duration_data.min(), duration_data.max(), 100)
    axes[1, 1].plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Corrélation
    corr = np.corrcoef(duration_data, burned_data)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    axes[1, 1].set_xlabel('Fire duration (timesteps)', fontsize=12)
    axes[1, 1].set_ylabel('Burned area (%)', fontsize=12)
    axes[1, 1].set_title('Correlation: Duration vs Burned area', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Afficher les statistiques
    print("\n=== Monte-Carlo Statistics ===")
    print(f"Number of simulations: {results['n_simulations']}")
    print(f"\nBurned area (%):")
    print(f"  Mean:   {results['burned']['mean']:.2f}")
    print(f"  Std:    {results['burned']['std']:.2f}")
    print(f"  Median: {results['burned']['median']:.2f}")
    print(f"  Range:  [{results['burned']['min']:.2f}, {results['burned']['max']:.2f}]")
    print(f"  IQR:    [{results['burned']['q25']:.2f}, {results['burned']['q75']:.2f}]")
    print(f"\nFire duration (timesteps):")
    print(f"  Mean:   {results['duration']['mean']:.2f}")
    print(f"  Std:    {results['duration']['std']:.2f}")
    print(f"  Median: {results['duration']['median']:.2f}")
    print(f"  Range:  [{results['duration']['min']:.0f}, {results['duration']['max']:.0f}]")
    
    return fig
