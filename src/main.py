"""
main.py - Script de démonstration du modèle de percolation
"""

import numpy as np
import matplotlib.pyplot as plt
from forest import ForestGrid
from visualization import (plot_grid, plot_evolution, animate_simulation,
                           plot_phase_diagram, compare_neighborhoods)
from analysis import (find_percolation_threshold, plot_percolation_curve,
                     analyze_density_effect, plot_density_analysis,
                     cluster_analysis, plot_cluster_distribution,
                     monte_carlo_statistics, plot_monte_carlo)

def demo_simple_simulation():
    """Démonstration 1: Simulation simple avec visualisation."""
    print("\n" + "="*60)
    print("DEMO 1: Simple simulation")
    print("="*60)
    
    # Création de la forêt
    forest = ForestGrid(size=100, tree_density=0.6, seed=42)
    print(f"Created: {forest}")
    
    # Déclencher un feu au centre
    forest.ignite_center()
    print("Fire ignited at center")
    
    # Visualiser l'état initial
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_grid(forest, title="Initial state (t=0)", ax=axes[0])
    
    # Simuler 5 pas
    for _ in range(5):
        forest.propagate(p=0.5, neighborhood='von_neumann')
    plot_grid(forest, title=f"After 5 timesteps (t={forest.timestep})", ax=axes[1])
    
    # Simuler jusqu'à extinction
    forest.simulate(p=0.5, neighborhood='von_neumann')
    plot_grid(forest, title=f"Final state (t={forest.timestep})", ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/demo1_simple_simulation.png', dpi=150, bbox_inches='tight')
    print("Saved: results/demo1_simple_simulation.png")
    
    # Statistiques finales
    stats = forest.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Duration: {stats['duration']} timesteps")
    print(f"  Burned: {stats['burned_cells']} cells ({stats['burned_percentage']:.1f}%)")
    print(f"  Burned trees: {stats['burned_trees_percentage']:.1f}%")
    
    # Évolution temporelle
    fig = plot_evolution(forest)
    plt.savefig('results/demo1_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: results/demo1_evolution.png")
    
    plt.show()

def demo_animation():
    """Démonstration 2: Animation de la propagation."""
    print("\n" + "="*60)
    print("DEMO 2: Animation")
    print("="*60)
    
    forest = ForestGrid(size=80, tree_density=0.65, seed=123)
    forest.ignite_center()
    
    print("Creating animation... (this may take a moment)")
    anim = animate_simulation(forest, p=0.5, neighborhood='von_neumann',
                             interval=100, max_steps=200,
                             save_path='results/demo2_animation.gif')
    
    print("Animation complete!")
    print("Note: To view the animation in Jupyter, use:")
    print("  from IPython.display import HTML")
    print("  HTML(anim.to_html5_video())")
    
    plt.show()

def demo_percolation_threshold():
    """Démonstration 3: Recherche du seuil de percolation."""
    print("\n" + "="*60)
    print("DEMO 3: Percolation threshold")
    print("="*60)
    
    # Recherche du seuil
    results = find_percolation_threshold(
        size=100,
        p_range=np.linspace(0.0, 1.0, 21),
        n_simulations=30,
        neighborhood='von_neumann',
        threshold=0.5
    )
    
    # Visualisation
    fig = plot_percolation_curve(results)
    plt.savefig('results/demo3_percolation_curve.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo3_percolation_curve.png")
    
    plt.show()

def demo_density_analysis():
    """Démonstration 4: Effet de la densité d'arbres."""
    print("\n" + "="*60)
    print("DEMO 4: Density effect")
    print("="*60)
    
    results = analyze_density_effect(
        size=100,
        densities=np.linspace(0.2, 1.0, 9),
        p=0.5,
        n_simulations=30,
        neighborhood='von_neumann'
    )
    
    fig = plot_density_analysis(results)
    plt.savefig('results/demo4_density_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo4_density_analysis.png")
    
    plt.show()

def demo_phase_diagram():
    """Démonstration 5: Diagramme de phase."""
    print("\n" + "="*60)
    print("DEMO 5: Phase diagram")
    print("="*60)
    print("Warning: This computation is intensive and may take several minutes!")
    
    fig = plot_phase_diagram(
        size=80,
        densities=np.linspace(0.3, 1.0, 8),
        probabilities=np.linspace(0.2, 1.0, 8),
        n_simulations=10,
        neighborhood='von_neumann'
    )
    
    plt.savefig('results/demo5_phase_diagram.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo5_phase_diagram.png")
    
    plt.show()

def demo_neighborhood_comparison():
    """Démonstration 6: Comparaison des voisinages."""
    print("\n" + "="*60)
    print("DEMO 6: Neighborhood comparison")
    print("="*60)
    
    fig = compare_neighborhoods(
        size=100,
        tree_density=0.6,
        p=0.5,
        n_simulations=30
    )
    
    plt.savefig('results/demo6_neighborhoods.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo6_neighborhoods.png")
    
    plt.show()

def demo_cluster_analysis():
    """Démonstration 7: Analyse des clusters."""
    print("\n" + "="*60)
    print("DEMO 7: Cluster analysis")
    print("="*60)
    
    # Plusieurs simulations avec paramètres différents
    results_list = []
    labels = []
    
    for p in [0.3, 0.5, 0.7]:
        forest = ForestGrid(size=100, tree_density=0.6, seed=42)
        forest.ignite_center()
        forest.simulate(p=p, neighborhood='von_neumann')
        
        cluster_results = cluster_analysis(forest)
        results_list.append(cluster_results)
        labels.append(f'p={p}')
        
        print(f"\nResults for p={p}:")
        print(f"  Number of clusters: {cluster_results['n_clusters']}")
        print(f"  Mean cluster size: {cluster_results['mean_size']:.1f}")
        print(f"  Max cluster size: {cluster_results['max_size']}")
    
    fig = plot_cluster_distribution(results_list, labels)
    plt.savefig('results/demo7_clusters.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo7_clusters.png")
    
    plt.show()

def demo_monte_carlo():
    """Démonstration 8: Analyse Monte-Carlo."""
    print("\n" + "="*60)
    print("DEMO 8: Monte-Carlo analysis")
    print("="*60)
    print("Running 1000 simulations... (this will take a moment)")
    
    results = monte_carlo_statistics(
        size=100,
        tree_density=0.6,
        p=0.5,
        n_simulations=1000,
        neighborhood='von_neumann'
    )
    
    fig = plot_monte_carlo(results)
    plt.savefig('results/demo8_monte_carlo.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/demo8_monte_carlo.png")
    
    plt.show()

def run_all_demos():
    """Exécute toutes les démonstrations."""
    import os
    
    # Créer le dossier results s'il n'existe pas
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("FOREST FIRE SIMULATION - Complete Demo Suite")
    print("="*60)
    
    demos = [
        ("Simple simulation", demo_simple_simulation),
        ("Animation", demo_animation),
        ("Percolation threshold", demo_percolation_threshold),
        ("Density analysis", demo_density_analysis),
        ("Phase diagram", demo_phase_diagram),
        ("Neighborhood comparison", demo_neighborhood_comparison),
        ("Cluster analysis", demo_cluster_analysis),
        ("Monte-Carlo analysis", demo_monte_carlo),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos)+1}. Run all demos")
    print("  0. Exit")
    
    choice = input(f"\nChoose a demo (0-{len(demos)+1}): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            print("Exiting...")
            return
        elif choice == len(demos) + 1:
            for name, func in demos:
                print(f"\nRunning: {name}")
                func()
        elif 1 <= choice <= len(demos):
            demos[choice-1][1]()
        else:
            print("Invalid choice!")
    except ValueError:
        print("Invalid input!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

if __name__ == "__main__":
    # Configuration matplotlib
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10
    
    # Menu interactif
    run_all_demos()
    
    # Alternative: exécuter une démo spécifique
    # demo_simple_simulation()
    # demo_animation()
    # demo_percolation_threshold()
    # etc.
