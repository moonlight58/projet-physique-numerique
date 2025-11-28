from src.forest import ForestGrid
from src.visualization import animate_simulation
from IPython.display import HTML
import matplotlib.pyplot as plt

forest = ForestGrid(size=60, tree_density=0.65, seed=123)
forest.ignite_center()

anim = animate_simulation(forest, p=0.5, interval=100, max_steps=150)

# Dans Jupyter
HTML(anim.to_jshtml())

# Ou sauvegarder
anim.save('fire_animation.gif', writer='pillow', fps=10)
