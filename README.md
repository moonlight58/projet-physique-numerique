# projet-physique-numerique
# Projet : Simulation de feux de forêt 🔥🌲

Excellent choix ! C'est un sujet riche qui combine automates cellulaires, physique, probabilités et optimisation. Voici un plan détaillé pour ton projet.

---

## 📋 Plan d'approfondissement progressif

### **Niveau 1 : Base (modèle de percolation simple)**

#### Objectifs :
- Implémenter le modèle classique de percolation
- Comprendre les automates cellulaires
- Étudier le seuil critique

#### Points techniques :
1. **Grille 2D** : forêt représentée par une matrice (arbre/vide/feu/cendres)
2. **Règles simples** :
   - Un arbre prend feu si un voisin brûle (4 ou 8 voisins)
   - Probabilité de propagation `p` constante
   - Le feu s'éteint après 1 timestep
3. **Paramètres** :
   - Densité d'arbres `d`
   - Probabilité de propagation `p`
   - Taille de grille `N×N`
4. **Mesures** :
   - Surface totale brûlée
   - Temps de propagation
   - Seuil de percolation critique

#### Visualisation :
- Animation matplotlib de la propagation
- Graphique : surface brûlée vs densité
- Diagramme de phase (d, p)

---

### **Niveau 2 : Modèle physique réaliste**

#### Ajouts :
1. **Vent** :
   - Vecteur (direction, intensité)
   - Probabilité de propagation augmentée dans la direction du vent
   - Formule : `p_vent = p_base × (1 + k × cos(θ))` où θ = angle avec le vent

2. **Humidité** :
   - Chaque cellule a un niveau d'humidité `h ∈ [0,1]`
   - Probabilité ajustée : `p_effective = p × (1 - h)`
   - Évaporation progressive de l'humidité

3. **Topographie** :
   - Carte d'altitude (générée par bruit de Perlin)
   - Le feu monte plus vite (gravité)
   - Formule : `p_pente = p × (1 + k × sin(pente))`

4. **Types de végétation** :
   - Herbe (propagation rapide, brûle vite)
   - Arbustes (moyen)
   - Arbres (lent, haute température)
   - Chaque type a ses propres paramètres (inflammabilité, durée de combustion)

#### Équations :

**Probabilité de propagation totale** :
```
p_total = p_base × f_humidite × f_vent × f_pente × f_vegetation
```

**Température d'une cellule** :
```
T(t+1) = T(t) + α × (T_voisins - T(t)) + Q_combustion - β × T(t)
```
- α : diffusion thermique
- Q : chaleur dégagée si en feu
- β : refroidissement

---

### **Niveau 3 : Diffusion thermique (équation de la chaleur)**

#### Modèle continu :
Au lieu d'un modèle purement probabiliste, on ajoute une couche physique.

**Équation de la chaleur 2D** :
```
∂T/∂t = k × (∂²T/∂x² + ∂²T/∂y²) + Q(x,y,t)
```

#### Implémentation :
1. **Discrétisation (différences finies)** :
   ```
   T[i,j](t+Δt) = T[i,j](t) + k×Δt/Δx² × (
       T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4×T[i,j]
   ) + Q[i,j]×Δt
   ```

2. **Source de chaleur Q** :
   - Q = 0 si pas de feu
   - Q = Q_max si combustion active
   - Dépend du type de végétation

3. **Seuil d'ignition** :
   - Un arbre s'enflamme si T > T_ignition (≈ 300°C)
   - Maintien du feu si T > T_combustion

4. **Condition CFL** (stabilité) :
   ```
   Δt ≤ Δx² / (4k)
   ```

#### Couplage hybride :
- **Propagation thermique** : diffusion continue de la chaleur
- **Ignition** : seuil de température déclenche le feu
- **Combustible** : chaque cellule a une quantité de matière combustible qui diminue

---

### **Niveau 4 : Stratégies d'intervention**

#### Objectif :
Optimiser les stratégies de lutte contre l'incendie.

#### Méthodes implémentables :

1. **Coupe-feu** :
   - Créer des bandes sans végétation
   - Placement optimal (algorithme génétique, A*)
   - Contrainte : budget limité

2. **Largage d'eau** :
   - Augmente localement l'humidité
   - Rayon d'action limité
   - Nombre de largages limité

3. **Contre-feu** :
   - Brûler volontairement une zone pour créer une barrière
   - Risque de perte de contrôle

#### Optimisation :
- **Algorithme génétique** pour placement des coupe-feux
- **Recherche locale** pour stratégie de largage
- **Programmation dynamique** pour séquence d'actions
- **Q-learning** : apprentissage par renforcement (avancé)

#### Métriques d'évaluation :
```python
score = w1 × surface_sauvée - w2 × coût_intervention - w3 × risque
```

---

### **Niveau 5 : Analyse statistique et théorie**

#### Études à mener :

1. **Théorie de la percolation** :
   - Seuil critique `p_c` : transition de phase
   - Exposants critiques : `β`, `γ`, `ν`
   - Clusters : distribution de tailles (loi de puissance)

2. **Analyse de sensibilité** :
   - Variation de chaque paramètre
   - Diagrammes de bifurcation
   - Surface de réponse (plans d'expériences)

3. **Simulations Monte-Carlo** :
   - Répéter 1000+ simulations
   - Distributions statistiques (surface brûlée, temps)
   - Intervalles de confiance

4. **Modèle en loi de puissance** :
   - Fréquence des feux vs surface : `P(A) ∝ A^(-α)`
   - Comparaison avec données réelles (statistiques forestières)

5. **Auto-organisation critique** :
   - Modèle de Drossel-Schwabl (forêt auto-organisée)
   - Avalanches, criticalité

---

## 🎯 Structure du projet Git

```
feux-de-foret/
│
├── README.md                          # Présentation, installation, résultats
├── requirements.txt                   # Dépendances Python
├── .gitignore
│
├── docs/
│   ├── rapport.pdf                    # Rapport mathématique complet
│   ├── equations.md                   # Dérivations mathématiques
│   └── references.bib                 # Bibliographie
│
├── notebooks/
│   ├── 01_percolation_simple.ipynb
│   ├── 02_modele_physique.ipynb
│   ├── 03_diffusion_thermique.ipynb
│   ├── 04_optimisation.ipynb
│   └── 05_analyse_statistique.ipynb
│
├── src/
│   ├── __init__.py
│   ├── forest.py                      # Classe ForestGrid
│   ├── fire_models.py                 # Modèles de propagation
│   ├── physics.py                     # Équations physiques
│   ├── interventions.py               # Stratégies de lutte
│   ├── optimization.py                # Algorithmes d'optimisation
│   ├── visualization.py               # Animations, plots
│   └── utils.py                       # Outils divers
│
├── tests/
│   ├── test_forest.py
│   ├── test_physics.py
│   └── test_optimization.py
│
├── data/
│   ├── topography/                    # Cartes d'altitude
│   ├── real_fires/                    # Données réelles (optionnel)
│   └── results/                       # Résultats de simulations
│
├── results/
│   ├── animations/                    # GIF, MP4
│   ├── figures/                       # PNG, PDF
│   └── data/                          # CSV, JSON
│
└── scripts/
    ├── run_simulation.py
    ├── parameter_sweep.py
    └── generate_animations.py
```

---

## 💻 Exemple de code (structure de base)

```python
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class CellState(Enum):
    EMPTY = 0
    TREE = 1
    FIRE = 2
    ASH = 3

class ForestGrid:
    def __init__(self, size, tree_density=0.6):
        self.size = size
        self.grid = np.random.choice(
            [CellState.EMPTY.value, CellState.TREE.value],
            size=(size, size),
            p=[1-tree_density, tree_density]
        )
        self.temperature = np.zeros((size, size))
        self.humidity = np.random.uniform(0.3, 0.7, (size, size))
        self.elevation = self.generate_terrain()

    def generate_terrain(self):
        # Bruit de Perlin ou simple gradient
        x = np.linspace(0, 1, self.size)
        y = np.linspace(0, 1, self.size)
        X, Y = np.meshgrid(x, y)
        return np.sin(4*np.pi*X) * np.cos(4*np.pi*Y)

    def ignite(self, x, y):
        """Déclenche un feu à la position (x, y)"""
        if self.grid[x, y] == CellState.TREE.value:
            self.grid[x, y] = CellState.FIRE.value
            self.temperature[x, y] = 1000  # °C

    def propagate_simple(self, p=0.5):
        """Modèle de percolation simple"""
        new_grid = self.grid.copy()
        fire_cells = np.argwhere(self.grid == CellState.FIRE.value)

        for x, y in fire_cells:
            # 4-voisinage
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == CellState.TREE.value:
                        if np.random.random() < p:
                            new_grid[nx, ny] = CellState.FIRE.value

            # Le feu s'éteint
            new_grid[x, y] = CellState.ASH.value

        self.grid = new_grid
        return np.sum(self.grid == CellState.FIRE.value) > 0  # Continue?

    def propagate_physical(self, p_base=0.3, wind=(0,0), k_wind=0.5):
        """Modèle avec vent et topographie"""
        new_grid = self.grid.copy()
        fire_cells = np.argwhere(self.grid == CellState.FIRE.value)

        for x, y in fire_cells:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                           (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == CellState.TREE.value:
                        # Facteur de vent
                        angle = np.arctan2(dy, dx)
                        wind_angle = np.arctan2(wind[1], wind[0])
                        wind_factor = 1 + k_wind * np.cos(angle - wind_angle)

                        # Facteur de pente
                        slope = self.elevation[nx, ny] - self.elevation[x, y]
                        slope_factor = 1 + 0.5 * slope if slope > 0 else 1

                        # Facteur d'humidité
                        humidity_factor = 1 - self.humidity[nx, ny]

                        # Probabilité totale
                        p_total = p_base * wind_factor * slope_factor * humidity_factor
                        p_total = np.clip(p_total, 0, 1)

                        if np.random.random() < p_total:
                            new_grid[nx, ny] = CellState.FIRE.value

            new_grid[x, y] = CellState.ASH.value

        self.grid = new_grid
        return np.sum(self.grid == CellState.FIRE.value) > 0

    def diffuse_heat(self, k=0.1, dt=0.1, dx=1.0):
        """Équation de la chaleur (schéma explicite)"""
        laplacian = (
            np.roll(self.temperature, 1, axis=0) +
            np.roll(self.temperature, -1, axis=0) +
            np.roll(self.temperature, 1, axis=1) +
            np.roll(self.temperature, -1, axis=1) -
            4 * self.temperature
        ) / dx**2

        # Source de chaleur (combustion)
        Q = np.zeros_like(self.temperature)
        Q[self.grid == CellState.FIRE.value] = 5000  # W/m²

        # Mise à jour
        self.temperature += k * dt * laplacian + Q * dt

        # Refroidissement
        self.temperature *= 0.95

        # Ignition par température
        ignition_mask = (self.temperature > 300) & (self.grid == CellState.TREE.value)
        self.grid[ignition_mask] = CellState.FIRE.value
```

---

## 📊 Idées de visualisations

### 1. **Animation de la propagation**
```python
import matplotlib.animation as animation

def animate_fire(forest, n_steps=100):
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['white', 'green', 'red', 'gray']
    cmap = ListedColormap(colors)

    im = ax.imshow(forest.grid, cmap=cmap, vmin=0, vmax=3)

    def update(frame):
        forest.propagate_physical(wind=(1, 0))
        im.set_data(forest.grid)
        ax.set_title(f'Timestep: {frame}')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=n_steps,
                                   interval=100, blit=True)
    return anim
```

### 2. **Diagramme de phase**
- Axes : densité d'arbres vs probabilité de propagation
- Couleur : surface brûlée moyenne
- Identifier le seuil critique

### 3. **Cartes de chaleur (température)**
- Superposer température et état des cellules
- Visualiser la diffusion thermique

### 4. **Analyse statistique**
- Histogrammes : distribution des surfaces brûlées
- Boxplots : comparaison de stratégies d'intervention
- Courbes ROC : efficacité des coupe-feux

---

## 🔬 Questions de recherche intéressantes

1. **Existe-t-il un seuil critique de densité en dessous duquel le feu ne se propage jamais ?**

2. **Comment l'intensité du vent affecte-t-elle la forme du feu (elliptique) ?**

3. **Quelle est la stratégie optimale de placement de coupe-feux avec un budget limité ?**

4. **Le modèle avec diffusion thermique donne-t-il des résultats significativement différents du modèle probabiliste ?**

5. **Peut-on reproduire les lois de puissance observées dans les feux de forêt réels ?**

---

## 📚 Références utiles

### Articles scientifiques :
- Drossel & Schwabl (1992) - Self-organized criticality in forest-fire model
- Rothermel (1972) - Mathematical model for fire spread (référence historique)
- Finney (1998) - FARSITE: Fire Area Simulator

### Ressources en ligne :
- Nicky Case - "Simulating the World (in Emoji)" (excellent tutoriel interactif)
- NetLogo - Modèle Fire (code open source)

### Librairies Python :
- `numpy`, `scipy` - Calcul numérique
- `matplotlib`, `seaborn` - Visualisation
- `mesa` - Framework pour automates cellulaires
- `numba` - Accélération de code (JIT)
- `pygame` - Visualisation interactive temps réel

---

Tu veux que je t'aide à démarrer avec un code de base complet, ou tu préfères qu'on approfondisse un aspect particulier (physique, optimisation, stats) ?
