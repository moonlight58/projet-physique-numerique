import numpy as np
import matplotlib.pyplot as plt
from src.forest import ForestGrid

# Tester différentes probabilités
probabilities = [0.3, 0.5, 0.7, 0.9]
results = {p: [] for p in probabilities}

for p in probabilities:
    for i in range(20):
        forest = ForestGrid(size=80, tree_density=0.6, seed=i)
        forest.ignite_center()
        forest.simulate(p=p)
        
        stats = forest.get_statistics()
        results[p].append(stats['burned_percentage'])

# Visualisation
plt.figure(figsize=(10, 6))
plt.boxplot([results[p] for p in probabilities], 
           labels=[f'p={p}' for p in probabilities])
plt.ylabel('Surface brûlée (%)')
plt.title('Effet de la probabilité de propagation')
plt.grid(True, alpha=0.3)
plt.show()
