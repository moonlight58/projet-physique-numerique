"""
generate_animations.py - Génère des animations de feux de forêt
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.forest import ForestGrid
from src.visualization import animate_simulation


def create_comparison_animation(output_dir, size=80, density=0.65):
    """
    Crée une animation comparant différentes probabilités.
    """
    print("Création d'une animation de comparaison...")
    
    probabilities = [0.3, 0.5, 0.7]
    
    for p in probabilities:
        print(f"  Probabilité p={p}...")
        
        forest = ForestGrid(size=size, tree_density=density, seed=42)
        forest.ignite_center()
        
        output_path = output_dir / f'animation_p{int(p*10)}.gif'
        
        anim = animate_simulation(
            forest,
            p=p,
            neighborhood='von_neumann',
            interval=100,
            max_steps=150,
            save_path=str(output_path)
        )
        
        plt.close()


def create_neighborhood_comparison(output_dir, size=80, density=0.65, p=0.5):
    """
    Crée des animations comparant les voisinages.
    """
    print("Création d'animations comparant les voisinages...")
    
    neighborhoods = ['von_neumann', 'moore']
    
    for neighborhood in neighborhoods:
        print(f"  Voisinage: {neighborhood}...")
        
        forest = ForestGrid(size=size, tree_density=density, seed=42)
        forest.ignite_center()
        
        output_path = output_dir / f'animation_{neighborhood}.gif'
        
        anim = animate_simulation(
            forest,
            p=p,
            neighborhood=neighborhood,
            interval=100,
            max_steps=150,
            save_path=str(output_path)
        )
        
        plt.close()


def create_density_comparison(output_dir, size=80, p=0.5):
    """
    Crée des animations pour différentes densités.
    """
    print("Création d'animations pour différentes densités...")
    
    densities = [0.3, 0.6, 0.9]
    
    for d in densities:
        print(f"  Densité d={d}...")
        
        forest = ForestGrid(size=size, tree_density=d, seed=42)
        forest.ignite_center()
        
        output_path = output_dir / f'animation_d{int(d*10)}.gif'
        
        anim = animate_simulation(
            forest,
            p=p,
            neighborhood='von_neumann',
            interval=100,
            max_steps=150,
            save_path=str(output_path)
        )
        
        plt.close()


def create_ignition_comparison(output_dir, size=80, density=0.65, p=0.5):
    """
    Crée des animations pour différents points d'ignition.
    """
    print("Création d'animations pour différents points d'ignition...")
    
    ignitions = {
        'center': lambda f: f.ignite_center(),
        'corner': lambda f: f.ignite(0, 0),
        'edge': lambda f: f.ignite(size//2, 0),
        'random': lambda f: f.ignite_random(5)
    }
    
    for name, ignite_func in ignitions.items():
        print(f"  Ignition: {name}...")
        
        forest = ForestGrid(size=size, tree_density=density, seed=42)
        ignite_func(forest)
        
        output_path = output_dir / f'animation_ignition_{name}.gif'
        
        anim = animate_simulation(
            forest,
            p=p,
            neighborhood='von_neumann',
            interval=100,
            max_steps=150,
            save_path=str(output_path)
        )
        
        plt.close()


def create_high_quality_animation(output_dir, size=100, density=0.65, 
                                 p=0.5, max_steps=300):
    """
    Crée une animation haute qualité pour présentation.
    """
    print("Création d'une animation haute qualité...")
    
    forest = ForestGrid(size=size, tree_density=density, seed=123)
    forest.ignite_center()
    
    output_path = output_dir / 'animation_high_quality.gif'
    
    anim = animate_simulation(
        forest,
        p=p,
        neighborhood='von_neumann',
        interval=50,  # Plus rapide
        max_steps=max_steps,
        save_path=str(output_path)
    )
    
    plt.close()
    
    print(f"  Animation haute qualité sauvegardée: {output_path}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Génère des animations de feux de forêt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('animation_type', 
                       choices=['all', 'comparison', 'neighborhood', 
                               'density', 'ignition', 'high_quality'],
                       help='Type d\'animation à créer')
    parser.add_argument('-o', '--output', type=str, default='results/animations',
                       help='Dossier de sortie')
    parser.add_argument('-s', '--size', type=int, default=80,
                       help='Taille de la grille')
    parser.add_argument('-d', '--density', type=float, default=0.65,
                       help='Densité d\'arbres')
    parser.add_argument('-p', '--probability', type=float, default=0.5,
                       help='Probabilité de propagation')
    parser.add_argument('--max-steps', type=int, default=150,
                       help='Nombre maximum de timesteps')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GÉNÉRATION D'ANIMATIONS")
    print("="*60)
    print(f"Dossier de sortie: {output_dir}")
    print()
    
    # Créer les animations selon le type
    if args.animation_type in ['all', 'comparison']:
        create_comparison_animation(output_dir, args.size, args.density)
    
    if args.animation_type in ['all', 'neighborhood']:
        create_neighborhood_comparison(output_dir, args.size, 
                                      args.density, args.probability)
    
    if args.animation_type in ['all', 'density']:
        create_density_comparison(output_dir, args.size, args.probability)
    
    if args.animation_type in ['all', 'ignition']:
        create_ignition_comparison(output_dir, args.size, 
                                  args.density, args.probability)
    
    if args.animation_type in ['high_quality']:
        create_high_quality_animation(output_dir, size=100, 
                                     density=args.density,
                                     p=args.probability,
                                     max_steps=args.max_steps)
    
    print("\n" + "="*60)
    print("Terminé!")
    print(f"Animations sauvegardées dans: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
