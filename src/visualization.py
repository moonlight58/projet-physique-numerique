"""
visualization.py - Fonctions pour visualiser les simulations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from src.forest import CellState, ForestGrid

# Configuration des couleurs
COLORS = {
    CellState.EMPTY.value: '#8B7355',    # Marron (sol)
    CellState.TREE.value: '#228B22',     # Vert (arbre)
    CellState.FIRE.value: '#FF4500',     # Rouge-orange (feu)
    CellState.ASH.value: '#2F4F4F'       # Gris foncé (cendres)
}

COLOR_MAP = ListedColormap([COLORS[i] for i in range(4)])

def plot_grid(forest, title=None, ax=None, show_stats=True):
    """
    Affiche l'état actuel de la grille.
    
    Args:
        forest (ForestGrid): Instance de ForestGrid
        title (str): Titre du graphique
        ax: Axes matplotlib (créé si None)
        show_stats (bool): Afficher les statistiques
    
    Returns:
        ax: Axes matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(forest.grid, cmap=COLOR_MAP, vmin=0, vmax=3, 
                   interpolation='nearest')
    
    if title is None:
        stats = forest.get_statistics()
        title = (f"Timestep: {forest.timestep} | "
                f"Burned: {stats['burned_cells']} cells "
                f"({stats['burned_percentage']:.1f}%)")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if show_stats:
        # Légende
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.EMPTY.value], label='Empty'),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.TREE.value], label='Tree'),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.FIRE.value], label='Fire'),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.ASH.value], label='Ash')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1, 1), fontsize=12)
    
    return ax

def plot_evolution(forest):
    """
    Affiche l'évolution temporelle des différents états.
    
    Args:
        forest (ForestGrid): Instance de ForestGrid après simulation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Nombre de cellules par état
    timesteps = forest.history['timesteps']
    
    ax1.plot(timesteps, forest.history['trees'], label='Trees', 
             color=COLORS[CellState.TREE.value], linewidth=2)
    ax1.plot(timesteps, forest.history['fires'], label='Fires', 
             color=COLORS[CellState.FIRE.value], linewidth=2)
    ax1.plot(timesteps, forest.history['ashes'], label='Ashes', 
             color=COLORS[CellState.ASH.value], linewidth=2)
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Number of cells', fontsize=12)
    ax1.set_title('Evolution of cell states over time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Proportions (stacked area)
    total = np.array(forest.history['trees']) + \
            np.array(forest.history['fires']) + \
            np.array(forest.history['ashes']) + \
            np.array(forest.history['empty'])
    
    trees_pct = np.array(forest.history['trees']) / total * 100
    fires_pct = np.array(forest.history['fires']) / total * 100
    ashes_pct = np.array(forest.history['ashes']) / total * 100
    empty_pct = np.array(forest.history['empty']) / total * 100
    
    ax2.fill_between(timesteps, 0, empty_pct, 
                     color=COLORS[CellState.EMPTY.value], alpha=0.7, label='Empty')
    ax2.fill_between(timesteps, empty_pct, empty_pct + trees_pct, 
                     color=COLORS[CellState.TREE.value], alpha=0.7, label='Trees')
    ax2.fill_between(timesteps, empty_pct + trees_pct, 
                     empty_pct + trees_pct + ashes_pct,
                     color=COLORS[CellState.ASH.value], alpha=0.7, label='Ashes')
    ax2.fill_between(timesteps, empty_pct + trees_pct + ashes_pct, 100,
                     color=COLORS[CellState.FIRE.value], alpha=0.7, label='Fires')
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Proportion of cell states over time', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def animate_simulation(forest, p=0.5, neighborhood='von_neumann', 
                       interval=200, max_steps=500, save_path=None):
    """
    Crée une animation de la propagation du feu.
    
    Args:
        forest (ForestGrid): Instance de ForestGrid avec feu initial
        p (float): Probabilité de propagation
        neighborhood (str): Type de voisinage
        interval (int): Délai entre les frames (ms)
        max_steps (int): Nombre maximum de frames
        save_path (str): Chemin pour sauvegarder l'animation (None = afficher)
    
    Returns:
        animation.FuncAnimation: Objet animation
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(forest.grid, cmap=COLOR_MAP, vmin=0, vmax=3, 
                   interpolation='nearest')
    ax.axis('off')
    
    # Titre avec statistiques
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                    ha='center', fontsize=14, fontweight='bold')
    
    # Légende
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.EMPTY.value], label='Empty'),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.TREE.value], label='Tree'),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.FIRE.value], label='Fire'),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[CellState.ASH.value], label='Ash')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1, 1), fontsize=12)
    
    # Variable pour stocker si le feu continue
    fire_active = [True]
    
    def update(frame):
        if fire_active[0]:
            fire_active[0] = forest.propagate(p, neighborhood)
            im.set_data(forest.grid)
            
            stats = forest.get_statistics()
            title.set_text(f"Timestep: {forest.timestep} | "
                          f"Burned: {stats['burned_cells']} cells "
                          f"({stats['burned_percentage']:.1f}%) | "
                          f"Active fires: {stats['burned_cells'] - forest.history['ashes'][-2] if len(forest.history['ashes']) > 1 else 0}")
        
        return [im, title]
    
    anim = animation.FuncAnimation(fig, update, frames=max_steps,
                                   interval=interval, blit=True, repeat=False)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)
        print(f"Animation saved to {save_path}")
    
    plt.tight_layout()
    return anim

def plot_phase_diagram(size=100, densities=None, probabilities=None, 
                       n_simulations=10, neighborhood='von_neumann'):
    """
    Crée un diagramme de phase (densité vs probabilité de propagation).
    
    Args:
        size (int): Taille de la grille
        densities (array): Valeurs de densité à tester
        probabilities (array): Valeurs de probabilité à tester
        n_simulations (int): Nombre de simulations par point
        neighborhood (str): Type de voisinage
    
    Returns:
        fig: Figure matplotlib
    """
    if densities is None:
        densities = np.linspace(0.1, 1.0, 10)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1.0, 10)
    
    results = np.zeros((len(densities), len(probabilities)))
    
    print("Computing phase diagram...")
    for i, d in enumerate(densities):
        for j, p in enumerate(probabilities):
            burned_list = []
            
            for _ in range(n_simulations):
                forest = ForestGrid(size, tree_density=d)
                forest.ignite_center()
                forest.simulate(p=p, neighborhood=neighborhood)
                
                stats = forest.get_statistics()
                burned_list.append(stats['burned_percentage'])
            
            results[i, j] = np.mean(burned_list)
        
        print(f"Progress: {(i+1)/len(densities)*100:.1f}%")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(results, origin='lower', cmap='hot', aspect='auto',
                   extent=[probabilities[0], probabilities[-1], 
                          densities[0], densities[-1]])
    
    ax.set_xlabel('Propagation probability (p)', fontsize=14)
    ax.set_ylabel('Tree density (d)', fontsize=14)
    ax.set_title('Phase diagram: Average burned area (%)', 
                fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Burned area (%)', fontsize=12)
    
    # Contours pour voir les transitions
    contours = ax.contour(probabilities, densities, results, 
                         levels=[10, 30, 50, 70, 90], colors='cyan', 
                         linewidths=2, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=10)
    
    plt.tight_layout()
    return fig

def compare_neighborhoods(size=100, tree_density=0.6, p=0.5, n_simulations=30):
    """
    Compare l'effet du voisinage (4 vs 8 voisins).
    
    Args:
        size (int): Taille de la grille
        tree_density (float): Densité d'arbres
        p (float): Probabilité de propagation
        n_simulations (int): Nombre de simulations
    
    Returns:
        fig: Figure matplotlib
    """
    results = {'von_neumann': [], 'moore': []}
    
    for neighborhood in ['von_neumann', 'moore']:
        for _ in range(n_simulations):
            forest = ForestGrid(size, tree_density)
            forest.ignite_center()
            forest.simulate(p=p, neighborhood=neighborhood)
            
            stats = forest.get_statistics()
            results[neighborhood].append(stats['burned_percentage'])
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    ax1.boxplot([results['von_neumann'], results['moore']], 
                labels=['Von Neumann\n(4 neighbors)', 'Moore\n(8 neighbors)'])
    ax1.set_ylabel('Burned area (%)', fontsize=12)
    ax1.set_title('Comparison of neighborhoods', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Histogrammes
    ax2.hist(results['von_neumann'], bins=20, alpha=0.6, 
            label='Von Neumann', color='blue', edgecolor='black')
    ax2.hist(results['moore'], bins=20, alpha=0.6, 
            label='Moore', color='red', edgecolor='black')
    ax2.set_xlabel('Burned area (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of burned areas', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Statistiques
    print("\n=== Comparison Statistics ===")
    print(f"Von Neumann (4 neighbors):")
    print(f"  Mean: {np.mean(results['von_neumann']):.2f}%")
    print(f"  Std:  {np.std(results['von_neumann']):.2f}%")
    print(f"\nMoore (8 neighbors):")
    print(f"  Mean: {np.mean(results['moore']):.2f}%")
    print(f"  Std:  {np.std(results['moore']):.2f}%")
    
    return fig
