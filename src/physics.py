"""
physics.py - Physical models for fire propagation

This module implements realistic physical effects:
- Wind (direction and speed)
- Humidity
- Topography (elevation effects)
- Vegetation types
- Thermal diffusion
"""

import numpy as np
from enum import Enum


class VegetationType(Enum):
    """Different types of vegetation with different fire behaviors"""
    GRASS = 0      # Fast spread, low intensity
    SHRUB = 1      # Medium spread, medium intensity
    TREE = 2       # Slow spread, high intensity
    DENSE_FOREST = 3  # Very slow spread, very high intensity


class VegetationProperties:
    """Physical properties of different vegetation types"""
    
    PROPERTIES = {
        VegetationType.GRASS: {
            'base_flammability': 0.9,      # Easy to ignite
            'burn_duration': 2,             # Burns quickly (timesteps)
            'heat_release': 200,            # Low heat (°C)
            'fuel_density': 0.3,            # Low fuel
            'ignition_temp': 200            # Easy ignition (°C)
        },
        VegetationType.SHRUB: {
            'base_flammability': 0.7,
            'burn_duration': 4,
            'heat_release': 400,
            'fuel_density': 0.6,
            'ignition_temp': 250
        },
        VegetationType.TREE: {
            'base_flammability': 0.5,
            'burn_duration': 8,
            'heat_release': 600,
            'fuel_density': 0.9,
            'ignition_temp': 300
        },
        VegetationType.DENSE_FOREST: {
            'base_flammability': 0.4,
            'burn_duration': 12,
            'heat_release': 800,
            'fuel_density': 1.0,
            'ignition_temp': 350
        }
    }
    
    @classmethod
    def get_property(cls, veg_type, property_name):
        """Get a specific property for a vegetation type"""
        return cls.PROPERTIES[veg_type][property_name]


class WindModel:
    """
    Wind model affecting fire propagation.
    
    Wind increases propagation probability in its direction and
    decreases it against the wind.
    """
    
    def __init__(self, direction=0.0, speed=0.0):
        """
        Initialize wind model.
        
        Args:
            direction (float): Wind direction in radians (0 = East, π/2 = North)
            speed (float): Wind speed coefficient [0, 1]
        """
        self.direction = direction
        self.speed = speed
        
        # Wind vector components
        self.vx = speed * np.cos(direction)
        self.vy = speed * np.sin(direction)
    
    def get_wind_factor(self, dx, dy):
        """
        Calculate wind effect on propagation from (0,0) to (dx, dy).
        
        Args:
            dx, dy (int): Direction of propagation
        
        Returns:
            float: Multiplicative factor for propagation probability
        """
        if self.speed == 0:
            return 1.0
        
        # Normalize direction
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return 1.0
        
        # Dot product between wind and propagation direction
        alignment = (dx * self.vx + dy * self.vy) / dist
        
        # Wind factor: 1 + speed * alignment
        # Range: [1 - speed, 1 + speed]
        factor = 1.0 + self.speed * alignment
        
        return max(0.1, factor)  # Minimum 0.1 to allow some back-propagation
    
    def modify_probability(self, p_base, from_pos, to_pos):
        """
        Modify propagation probability based on wind.
        
        Args:
            p_base (float): Base propagation probability
            from_pos (tuple): (x, y) of fire source
            to_pos (tuple): (x, y) of target cell
        
        Returns:
            float: Modified probability
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        wind_factor = self.get_wind_factor(dx, dy)
        return min(1.0, p_base * wind_factor)
    
    def __repr__(self):
        angle_deg = np.degrees(self.direction) % 360
        return f"Wind(direction={angle_deg:.1f}°, speed={self.speed:.2f})"


class HumidityMap:
    """
    Spatial humidity distribution affecting fire propagation.
    
    Higher humidity reduces flammability and propagation probability.
    """
    
    def __init__(self, size, base_humidity=0.3, variation=0.2, seed=None):
        """
        Initialize humidity map.
        
        Args:
            size (int): Grid size
            base_humidity (float): Average humidity [0, 1]
            variation (float): Spatial variation
            seed (int): Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.size = size
        self.base_humidity = base_humidity
        
        # Generate spatial humidity variation using simple noise
        self.map = self._generate_humidity_field(base_humidity, variation)
    
    def _generate_humidity_field(self, base, variation):
        """Generate spatially correlated humidity field"""
        # Simple approach: smooth random field
        raw = np.random.randn(self.size, self.size) * variation + base
        
        # Smooth with Gaussian filter for spatial correlation
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(raw, sigma=self.size / 10)
        
        # Clip to [0, 1]
        return np.clip(smoothed, 0, 1)
    
    def get_humidity(self, x, y):
        """Get humidity at position (x, y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.map[x, y]
        return self.base_humidity
    
    def modify_probability(self, p_base, x, y):
        """
        Modify propagation probability based on humidity.
        
        High humidity reduces fire spread.
        
        Args:
            p_base (float): Base probability
            x, y (int): Position
        
        Returns:
            float: Modified probability
        """
        h = self.get_humidity(x, y)
        # Linear decrease with humidity
        return p_base * (1 - 0.8 * h)  # Max 80% reduction
    
    def increase_humidity(self, center, radius, amount=0.3):
        """
        Increase humidity in an area (e.g., water drop).
        
        Args:
            center (tuple): (x, y) center position
            radius (float): Effect radius
            amount (float): Humidity increase
        """
        x0, y0 = center
        
        for x in range(max(0, x0 - radius), min(self.size, x0 + radius + 1)):
            for y in range(max(0, y0 - radius), min(self.size, y0 + radius + 1)):
                dist = np.sqrt((x - x0)**2 + (y - y0)**2)
                if dist <= radius:
                    # Gaussian-like falloff
                    factor = np.exp(-dist**2 / (2 * (radius/2)**2))
                    self.map[x, y] = min(1.0, self.map[x, y] + amount * factor)
    
    def evaporate(self, rate=0.01):
        """
        Simulate humidity evaporation over time.
        
        Args:
            rate (float): Evaporation rate per timestep
        """
        self.map = np.maximum(self.base_humidity, self.map - rate)


class TopographyMap:
    """
    Elevation map affecting fire propagation.
    
    Fire spreads faster uphill due to convection.
    """
    
    def __init__(self, size, max_elevation=100, seed=None):
        """
        Initialize topography map.
        
        Args:
            size (int): Grid size
            max_elevation (float): Maximum elevation difference
            seed (int): Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.size = size
        self.max_elevation = max_elevation
        
        # Generate elevation map
        self.elevation = self._generate_terrain()
    
    def _generate_terrain(self):
        """Generate realistic terrain using Perlin-like noise"""
        from scipy.ndimage import gaussian_filter
        
        # Multi-scale noise for realistic terrain
        terrain = np.zeros((self.size, self.size))
        
        # Large features
        terrain += gaussian_filter(np.random.randn(self.size, self.size), 
                                   sigma=self.size / 4)
        
        # Medium features
        terrain += 0.5 * gaussian_filter(np.random.randn(self.size, self.size), 
                                         sigma=self.size / 10)
        
        # Small features
        terrain += 0.25 * gaussian_filter(np.random.randn(self.size, self.size), 
                                          sigma=self.size / 20)
        
        # Normalize to [0, max_elevation]
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        terrain *= self.max_elevation
        
        return terrain
    
    def get_elevation(self, x, y):
        """Get elevation at position (x, y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.elevation[x, y]
        return 0
    
    def get_slope(self, from_pos, to_pos):
        """
        Calculate slope between two positions.
        
        Args:
            from_pos (tuple): (x, y) source position
            to_pos (tuple): (x, y) target position
        
        Returns:
            float: Slope in radians (positive = uphill)
        """
        z1 = self.get_elevation(*from_pos)
        z2 = self.get_elevation(*to_pos)
        
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        
        if horizontal_dist == 0:
            return 0
        
        return np.arctan((z2 - z1) / horizontal_dist)
    
    def modify_probability(self, p_base, from_pos, to_pos, slope_factor=2.0):
        """
        Modify propagation probability based on slope.
        
        Fire spreads faster uphill.
        
        Args:
            p_base (float): Base probability
            from_pos (tuple): Source position
            to_pos (tuple): Target position
            slope_factor (float): How much slope affects spread
        
        Returns:
            float: Modified probability
        """
        slope = self.get_slope(from_pos, to_pos)
        
        # Uphill increases probability, downhill decreases it
        # Factor: exp(slope_factor * tan(slope))
        factor = np.exp(slope_factor * np.tan(slope))
        
        return min(1.0, p_base * factor)


class ThermalDiffusion:
    """
    Heat diffusion model using the heat equation.
    
    Implements: ∂T/∂t = k·∇²T
    """
    
    def __init__(self, size, k=0.1, dt=0.1, dx=1.0):
        """
        Initialize thermal diffusion.
        
        Args:
            size (int): Grid size
            k (float): Thermal diffusivity
            dt (float): Time step
            dx (float): Spatial step
        """
        self.size = size
        self.k = k
        self.dt = dt
        self.dx = dx
        
        # Check CFL stability condition
        cfl = k * dt / (dx ** 2)
        if cfl > 0.25:
            raise ValueError(f"CFL condition violated: {cfl:.3f} > 0.25. "
                           f"Reduce dt or increase dx.")
        
        # Temperature field (°C)
        self.temperature = np.ones((size, size)) * 20  # Ambient
    
    def diffuse(self):
        """
        Apply one timestep of heat diffusion.
        
        Returns:
            np.ndarray: Updated temperature field
        """
        T = self.temperature
        
        # 5-point stencil Laplacian
        laplacian = (
            np.roll(T, 1, axis=0) +   # T[i-1,j]
            np.roll(T, -1, axis=0) +  # T[i+1,j]
            np.roll(T, 1, axis=1) +   # T[i,j-1]
            np.roll(T, -1, axis=1) -  # T[i,j+1]
            4 * T                      # -4*T[i,j]
        ) / (self.dx ** 2)
        
        # Update: T_new = T_old + k*dt*∇²T
        new_T = T + self.k * self.dt * laplacian
        
        # Neumann boundary conditions (∂T/∂n = 0)
        new_T[0, :] = new_T[1, :]
        new_T[-1, :] = new_T[-2, :]
        new_T[:, 0] = new_T[:, 1]
        new_T[:, -1] = new_T[:, -2]
        
        self.temperature = new_T
        return new_T
    
    def add_heat_source(self, position, heat):
        """Add heat at a specific position"""
        x, y = position
        if 0 <= x < self.size and 0 <= y < self.size:
            self.temperature[x, y] += heat * self.dt
    
    def cool_to_ambient(self, ambient_temp=20, cooling_rate=0.05):
        """Apply natural cooling"""
        self.temperature -= cooling_rate * (self.temperature - ambient_temp) * self.dt


class PhysicalForestModel:
    """
    Complete physical model combining all effects.
    
    This integrates:
    - Wind
    - Humidity
    - Topography
    - Vegetation types
    - Thermal diffusion (optional)
    """
    
    def __init__(self, forest, wind=None, humidity=None, topography=None,
                 vegetation=None, use_thermal=False):
        """
        Initialize physical model.
        
        Args:
            forest: ForestGrid instance
            wind: WindModel instance
            humidity: HumidityMap instance
            topography: TopographyMap instance
            vegetation: 2D array of VegetationType (optional)
            use_thermal: Whether to use thermal diffusion
        """
        self.forest = forest
        self.wind = wind or WindModel(0, 0)
        self.humidity = humidity or HumidityMap(forest.size)
        self.topography = topography or TopographyMap(forest.size)
        self.vegetation = vegetation
        
        self.use_thermal = use_thermal
        if use_thermal:
            self.thermal = ThermalDiffusion(forest.size)
    
    def calculate_propagation_probability(self, p_base, from_pos, to_pos):
        """
        Calculate effective propagation probability with all physical effects.
        
        Args:
            p_base (float): Base probability
            from_pos (tuple): (x, y) source
            to_pos (tuple): (x, y) target
        
        Returns:
            float: Modified probability
        """
        p = p_base
        
        # Apply wind effect
        p = self.wind.modify_probability(p, from_pos, to_pos)
        
        # Apply humidity effect
        p = self.humidity.modify_probability(p, to_pos[0], to_pos[1])
        
        # Apply topography effect
        p = self.topography.modify_probability(p, from_pos, to_pos)
        
        # Apply vegetation effect if available
        if self.vegetation is not None:
            x, y = to_pos
            if 0 <= x < self.forest.size and 0 <= y < self.forest.size:
                veg_type = self.vegetation[x, y]
                flammability = VegetationProperties.get_property(
                    veg_type, 'base_flammability'
                )
                p *= flammability
        
        return np.clip(p, 0, 1)
    
    def propagate_physical(self, p_base=0.5, neighborhood='von_neumann'):
        """
        Propagate fire with physical effects.
        
        Args:
            p_base (float): Base propagation probability
            neighborhood (str): Neighborhood type
        
        Returns:
            bool: True if fire continues
        """
        from forest import CellState
        
        new_grid = self.forest.grid.copy()
        fire_cells = np.argwhere(self.forest.grid == CellState.FIRE.value)
        
        # Propagate from each fire cell
        for fx, fy in fire_cells:
            neighbors = self.forest.get_neighbors(fx, fy, neighborhood)
            
            for nx, ny in neighbors:
                if self.forest.grid[nx, ny] == CellState.TREE.value:
                    # Calculate physical probability
                    p_eff = self.calculate_propagation_probability(
                        p_base, (fx, fy), (nx, ny)
                    )
                    
                    # Stochastic ignition
                    if np.random.random() < p_eff:
                        new_grid[nx, ny] = CellState.FIRE.value
            
            # Fire burns out
            new_grid[fx, fy] = CellState.ASH.value
        
        self.forest.grid = new_grid
        self.forest.timestep += 1
        self.forest._record_state()
        
        # Apply environmental changes
        self.humidity.evaporate(rate=0.005)
        
        return np.sum(self.forest.grid == CellState.FIRE.value) > 0


# Utility functions
def create_wind_from_direction(direction_name, speed=0.5):
    """
    Create wind from cardinal direction name.
    
    Args:
        direction_name (str): 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
        speed (float): Wind speed
    
    Returns:
        WindModel
    """
    directions = {
        'E': 0,
        'NE': np.pi/4,
        'N': np.pi/2,
        'NW': 3*np.pi/4,
        'W': np.pi,
        'SW': 5*np.pi/4,
        'S': 3*np.pi/2,
        'SE': 7*np.pi/4
    }
    
    angle = directions.get(direction_name.upper(), 0)
    return WindModel(angle, speed)


def generate_vegetation_map(size, seed=None):
    """
    Generate a random vegetation map.
    
    Args:
        size (int): Grid size
        seed (int): Random seed
    
    Returns:
        np.ndarray: Array of VegetationType
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create patches of different vegetation
    veg_map = np.random.choice(
        [VegetationType.GRASS, VegetationType.SHRUB, 
         VegetationType.TREE, VegetationType.DENSE_FOREST],
        size=(size, size),
        p=[0.3, 0.3, 0.3, 0.1]  # More grass and shrubs
    )
    
    # Smooth to create patches
    from scipy.ndimage import gaussian_filter
    
    # Convert to numeric for smoothing
    numeric = np.array([[v.value for v in row] for row in veg_map])
    smoothed = gaussian_filter(numeric.astype(float), sigma=size/20)
    
    # Convert back to VegetationType
    veg_map = np.array([[
        VegetationType(int(round(smoothed[i, j])))
        for j in range(size)
    ] for i in range(size)])
    
    return veg_map


# Example usage
if __name__ == "__main__":
    from forest import ForestGrid
    import matplotlib.pyplot as plt
    
    # Create forest
    forest = ForestGrid(size=100, tree_density=0.7, seed=42)
    
    # Create physical effects
    wind = create_wind_from_direction('NE', speed=0.6)
    humidity = HumidityMap(100, base_humidity=0.3, seed=42)
    topography = TopographyMap(100, max_elevation=50, seed=42)
    
    print(f"Wind: {wind}")
    print(f"Average humidity: {humidity.map.mean():.2f}")
    print(f"Elevation range: [{topography.elevation.min():.1f}, "
          f"{topography.elevation.max():.1f}]")
    
    # Create physical model
    physical_model = PhysicalForestModel(
        forest, wind=wind, humidity=humidity, topography=topography
    )
    
    # Ignite and simulate
    forest.ignite_center()
    
    print("\nSimulating with physical effects...")
    steps = 0
    max_steps = 200
    
    while physical_model.propagate_physical(p_base=0.5) and steps < max_steps:
        steps += 1
        if steps % 20 == 0:
            print(f"Step {steps}: {np.sum(forest.grid == 2)} fires active")
    
    print(f"\nSimulation complete after {steps} steps")
    
    stats = forest.get_statistics()
    print(f"Burned: {stats['burned_cells']} cells ({stats['burned_percentage']:.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Forest state
    from visualization import plot_grid
    plot_grid(forest, ax=axes[0, 0])
    
    # Humidity
    im1 = axes[0, 1].imshow(humidity.map, cmap='Blues')
    axes[0, 1].set_title('Humidity Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Topography
    im2 = axes[1, 0].imshow(topography.elevation, cmap='terrain')
    axes[1, 0].set_title('Topography')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Wind visualization
    axes[1, 1].arrow(50, 50, 
                     30 * np.cos(wind.direction), 
                     30 * np.sin(wind.direction),
                     head_width=5, head_length=5, fc='red', ec='red')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title(f'Wind: {wind}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('physical_simulation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: physical_simulation.png")
    
    plt.show()
