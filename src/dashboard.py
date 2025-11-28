"""
dashboard.py - Interactive visualization dashboard for physical fire simulation

Run with: python dashboard.py

This creates an interactive matplotlib dashboard with sliders and controls
to visualize all physical effects in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import ListedColormap
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from forest import ForestGrid, CellState
from physics import (PhysicalForestModel, WindModel, HumidityMap, 
                     TopographyMap, create_wind_from_direction,
                     VegetationType, generate_vegetation_map)


class PhysicsDashboard:
    """Interactive dashboard for physical fire simulation"""
    
    def __init__(self, size=80):
        self.size = size
        self.seed = 42
        
        # Initialize simulation
        self.reset_simulation()
        
        # Simulation state
        self.running = False
        self.step_count = 0
        
        # Track colorbars to avoid duplication
        self.colorbars = {}
        
        # Setup figure and axes
        self.setup_figure()
        
        # Setup controls (must be after setup_figure)
        self.setup_controls()
        
        # Now do initial update
        self.update_plots()
        
    def reset_simulation(self):
        """Reset the simulation with current parameters"""
        # Create forest
        self.forest = ForestGrid(
            size=self.size, 
            tree_density=0.7, 
            seed=self.seed
        )
        
        # Physical effects (will be updated by sliders)
        self.wind = WindModel(direction=np.pi/4, speed=0.5)
        self.humidity = HumidityMap(self.size, base_humidity=0.3, seed=self.seed)
        self.topography = TopographyMap(self.size, max_elevation=50, seed=self.seed)
        self.vegetation = None  # Optional
        
        # Physical model
        self.physical_model = PhysicalForestModel(
            self.forest,
            wind=self.wind,
            humidity=self.humidity,
            topography=self.topography,
            vegetation=self.vegetation
        )
        
        # Ignite center
        self.forest.ignite_center()
        
        self.step_count = 0
    
    def setup_figure(self):
        """Setup the matplotlib figure with subplots"""
        self.fig = plt.figure(figsize=(18, 12))
        
        # Grid layout
        gs = self.fig.add_gridspec(3, 4, left=0.05, right=0.95, 
                                   bottom=0.25, top=0.95,
                                   hspace=0.3, wspace=0.3)
        
        # Main forest view (larger)
        self.ax_forest = self.fig.add_subplot(gs[0:2, 0:2])
        
        # Physical layers
        self.ax_humidity = self.fig.add_subplot(gs[0, 2])
        self.ax_topo = self.fig.add_subplot(gs[0, 3])
        self.ax_wind = self.fig.add_subplot(gs[1, 2])
        self.ax_stats = self.fig.add_subplot(gs[1, 3])
        
        # Evolution plots
        self.ax_evolution = self.fig.add_subplot(gs[2, :2])
        self.ax_temp = self.fig.add_subplot(gs[2, 2:])
        
        # Color map for forest
        colors = ['white', 'green', 'red', 'gray']
        self.forest_cmap = ListedColormap(colors)
    
    def setup_controls(self):
        """Setup interactive controls (sliders, buttons)"""
        # Control area at bottom
        control_height = 0.20
        control_left = 0.05
        control_width = 0.90
        
        slider_height = 0.02
        slider_spacing = 0.03
        y_start = 0.18
        
        # Wind direction slider
        ax_wind_dir = plt.axes([control_left, y_start, control_width, slider_height])
        self.slider_wind_dir = Slider(
            ax_wind_dir, 'Wind Direction (°)', 
            0, 360, valinit=45, valstep=15
        )
        self.slider_wind_dir.on_changed(self.update_wind)
        
        # Wind speed slider
        y_start -= slider_spacing
        ax_wind_speed = plt.axes([control_left, y_start, control_width, slider_height])
        self.slider_wind_speed = Slider(
            ax_wind_speed, 'Wind Speed', 
            0, 1.0, valinit=0.5, valstep=0.1
        )
        self.slider_wind_speed.on_changed(self.update_wind)
        
        # Base probability slider
        y_start -= slider_spacing
        ax_prob = plt.axes([control_left, y_start, control_width, slider_height])
        self.slider_prob = Slider(
            ax_prob, 'Base Probability', 
            0, 1.0, valinit=0.5, valstep=0.05
        )
        
        # Humidity slider
        y_start -= slider_spacing
        ax_humidity = plt.axes([control_left, y_start, control_width, slider_height])
        self.slider_humidity = Slider(
            ax_humidity, 'Base Humidity', 
            0, 1.0, valinit=0.3, valstep=0.1
        )
        self.slider_humidity.on_changed(self.update_humidity)
        
        # Buttons
        button_width = 0.1
        button_height = 0.04
        button_spacing = 0.12
        button_y = 0.02
        
        # Step button
        ax_step = plt.axes([control_left, button_y, button_width, button_height])
        self.btn_step = Button(ax_step, 'Step')
        self.btn_step.on_clicked(self.step_simulation)
        
        # Run/Pause button
        ax_run = plt.axes([control_left + button_spacing, button_y, 
                          button_width, button_height])
        self.btn_run = Button(ax_run, 'Run')
        self.btn_run.on_clicked(self.toggle_run)
        
        # Reset button
        ax_reset = plt.axes([control_left + 2*button_spacing, button_y, 
                            button_width, button_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)
        
        # Water drop button
        ax_water = plt.axes([control_left + 3*button_spacing, button_y, 
                            button_width, button_height])
        self.btn_water = Button(ax_water, 'Water Drop')
        self.btn_water.on_clicked(self.water_drop)
        
        # Info text
        self.info_text = self.fig.text(
            0.75, 0.02, '', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    def update_wind(self, val=None):
        """Update wind parameters"""
        direction_deg = self.slider_wind_dir.val
        speed = self.slider_wind_speed.val
        
        direction_rad = np.radians(direction_deg)
        self.wind = WindModel(direction_rad, speed)
        self.physical_model.wind = self.wind
        
        self.update_plots()
    
    def update_humidity(self, val=None):
        """Update humidity parameters"""
        base_humidity = self.slider_humidity.val
        self.humidity = HumidityMap(self.size, base_humidity=base_humidity, 
                                     seed=self.seed)
        self.physical_model.humidity = self.humidity
        
        self.update_plots()
    
    def update_plots(self):
        """Redraw all plots"""
        # Clear axes
        for ax in [self.ax_forest, self.ax_humidity, self.ax_topo, 
                   self.ax_wind, self.ax_stats, self.ax_evolution, self.ax_temp]:
            ax.clear()
        
        # 1. Forest state
        im_forest = self.ax_forest.imshow(
            self.forest.grid, 
            cmap=self.forest_cmap, 
            vmin=0, vmax=3,
            interpolation='nearest'
        )
        stats = self.forest.get_statistics()
        self.ax_forest.set_title(
            f'Forest State (t={self.step_count})\n'
            f'Burned: {stats["burned_percentage"]:.1f}%',
            fontsize=12, fontweight='bold'
        )
        self.ax_forest.axis('off')
        
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='white', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, fc='green', label='Tree'),
            plt.Rectangle((0, 0), 1, 1, fc='red', label='Fire'),
            plt.Rectangle((0, 0), 1, 1, fc='gray', label='Ash')
        ]
        self.ax_forest.legend(handles=legend_elements, loc='upper left',
                             bbox_to_anchor=(1, 1), fontsize=9)
        
        # 2. Humidity map
        im_hum = self.ax_humidity.imshow(self.humidity.map, cmap='Blues', 
                                         vmin=0, vmax=1)
        self.ax_humidity.set_title('Humidity', fontsize=10, fontweight='bold')
        self.ax_humidity.axis('off')
        
        # Add colorbar only once
        if 'humidity' not in self.colorbars:
            self.colorbars['humidity'] = plt.colorbar(im_hum, ax=self.ax_humidity, fraction=0.046)
        else:
            self.colorbars['humidity'].update_normal(im_hum)
        
        # 3. Topography
        im_topo = self.ax_topo.imshow(self.topography.elevation, cmap='terrain')
        self.ax_topo.set_title('Elevation (m)', fontsize=10, fontweight='bold')
        self.ax_topo.axis('off')
        
        # Add colorbar only once
        if 'topography' not in self.colorbars:
            self.colorbars['topography'] = plt.colorbar(im_topo, ax=self.ax_topo, fraction=0.046)
        else:
            self.colorbars['topography'].update_normal(im_topo)
        
        # 4. Wind visualization
        self.ax_wind.set_xlim(-1, 1)
        self.ax_wind.set_ylim(-1, 1)
        self.ax_wind.set_aspect('equal')
        
        # Draw wind arrow
        arrow_scale = 0.7
        dx = arrow_scale * np.cos(self.wind.direction)
        dy = arrow_scale * np.sin(self.wind.direction)
        
        self.ax_wind.arrow(0, 0, dx, dy, 
                          head_width=0.15, head_length=0.15,
                          fc='red', ec='darkred', linewidth=2)
        
        # Wind speed circle
        circle = plt.Circle((0, 0), self.wind.speed, 
                           fill=False, linestyle='--', 
                           color='gray', alpha=0.5)
        self.ax_wind.add_patch(circle)
        
        # Cardinal directions
        for angle, label in [(0, 'E'), (90, 'N'), (180, 'W'), (270, 'S')]:
            rad = np.radians(angle)
            self.ax_wind.text(0.85*np.cos(rad), 0.85*np.sin(rad), label,
                            ha='center', va='center', fontsize=8, color='gray')
        
        self.ax_wind.set_title(f'Wind\nSpeed: {self.wind.speed:.2f}', 
                              fontsize=10, fontweight='bold')
        self.ax_wind.grid(True, alpha=0.3)
        self.ax_wind.set_xticks([])
        self.ax_wind.set_yticks([])
        
        # 5. Statistics
        self.ax_stats.axis('off')
        
        stats_text = (
            f"Statistics:\n\n"
            f"Timestep: {self.step_count}\n"
            f"Total cells: {stats['total_cells']}\n"
            f"Initial trees: {stats['initial_trees']}\n"
            f"Burned: {stats['burned_cells']}\n"
            f"Burned %: {stats['burned_percentage']:.1f}%\n"
            f"Remaining trees: {stats['final_trees']}\n"
            f"Active fires: {np.sum(self.forest.grid == CellState.FIRE.value)}\n\n"
            f"Physical Parameters:\n"
            f"Wind dir: {np.degrees(self.wind.direction):.0f}°\n"
            f"Wind speed: {self.wind.speed:.2f}\n"
            f"Avg humidity: {self.humidity.map.mean():.2f}\n"
            f"Elevation range: {self.topography.elevation.min():.0f}-"
            f"{self.topography.elevation.max():.0f}m"
        )
        
        self.ax_stats.text(0.05, 0.95, stats_text, 
                          transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=9,
                          family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 6. Evolution plot
        if len(self.forest.history['timesteps']) > 1:
            timesteps = self.forest.history['timesteps']
            
            self.ax_evolution.plot(timesteps, self.forest.history['trees'], 
                                  label='Trees', color='green', linewidth=2)
            self.ax_evolution.plot(timesteps, self.forest.history['fires'], 
                                  label='Fires', color='red', linewidth=2)
            self.ax_evolution.plot(timesteps, self.forest.history['ashes'], 
                                  label='Ashes', color='gray', linewidth=2)
            
            self.ax_evolution.set_xlabel('Timestep', fontsize=10)
            self.ax_evolution.set_ylabel('Number of cells', fontsize=10)
            self.ax_evolution.set_title('Evolution over time', 
                                       fontsize=10, fontweight='bold')
            self.ax_evolution.legend(fontsize=9)
            self.ax_evolution.grid(True, alpha=0.3)
        
        # 7. Temperature distribution (if available)
        if hasattr(self.physical_model, 'thermal') and self.physical_model.use_thermal:
            im_temp = self.ax_temp.imshow(self.physical_model.thermal.temperature,
                                         cmap='hot', vmin=0, vmax=500)
            self.ax_temp.set_title('Temperature (°C)', fontsize=10, fontweight='bold')
            self.ax_temp.axis('on')
            
            # Add colorbar only once
            if 'temperature' not in self.colorbars:
                self.colorbars['temperature'] = plt.colorbar(im_temp, ax=self.ax_temp, fraction=0.046)
            else:
                self.colorbars['temperature'].update_normal(im_temp)
        else:
            # Show burned percentage over time
            if len(self.forest.history['timesteps']) > 1:
                timesteps = self.forest.history['timesteps']
                total = self.forest.size ** 2
                burned_pct = [ash/total*100 for ash in self.forest.history['ashes']]
                
                self.ax_temp.plot(timesteps, burned_pct, 'r-', linewidth=2)
                self.ax_temp.set_xlabel('Timestep', fontsize=10)
                self.ax_temp.set_ylabel('Burned area (%)', fontsize=10)
                self.ax_temp.set_title('Cumulative burned area', 
                                      fontsize=10, fontweight='bold')
                self.ax_temp.grid(True, alpha=0.3)
                self.ax_temp.set_ylim(0, 100)
            
            self.ax_temp.axis('on')
        
        # Update info text
        active_fires = np.sum(self.forest.grid == CellState.FIRE.value)
        if active_fires > 0:
            status = f"ACTIVE - {active_fires} fires burning"
            color = 'orange'
        else:
            status = "EXTINGUISHED"
            color = 'lightgreen'
        
        if hasattr(self, 'info_text'):
            self.info_text.set_text(status)
            self.info_text.set_bbox(dict(boxstyle='round', facecolor=color, alpha=0.5))
        
        self.fig.canvas.draw_idle()
    
    def step_simulation(self, event=None):
        """Advance simulation by one step"""
        p_base = self.slider_prob.val
        continues = self.physical_model.propagate_physical(p_base=p_base)
        self.step_count += 1
        
        self.update_plots()
        
        if not continues and self.running:
            self.running = False
            self.btn_run.label.set_text('Run')
    
    def toggle_run(self, event):
        """Toggle run/pause"""
        self.running = not self.running
        
        if self.running:
            self.btn_run.label.set_text('Pause')
            self.run_simulation()
        else:
            self.btn_run.label.set_text('Run')
    
    def run_simulation(self):
        """Run simulation continuously"""
        if self.running:
            active = np.sum(self.forest.grid == CellState.FIRE.value) > 0
            
            if active:
                self.step_simulation()
                self.fig.canvas.draw_idle()
                # Schedule next step
                self.fig.canvas.manager.window.after(100, self.run_simulation)
            else:
                self.running = False
                self.btn_run.label.set_text('Run')
    
    def reset(self, event):
        """Reset simulation"""
        self.running = False
        self.btn_run.label.set_text('Run')
        
        # Clear colorbars for fresh start
        self.colorbars = {}
        
        self.reset_simulation()
        self.update_plots()
    
    def water_drop(self, event):
        """Simulate water drop at center"""
        center = (self.size // 2, self.size // 2)
        radius = int(self.size * 0.1)  # 10% of grid size
        
        self.humidity.increase_humidity(center, radius, amount=0.4)
        self.update_plots()
        
        print(f"Water drop at {center} with radius {radius}")
    
    def show(self):
        """Display the dashboard"""
        plt.show()


def main():
    """Main function to run the dashboard"""
    print("="*60)
    print("FOREST FIRE PHYSICS DASHBOARD")
    print("="*60)
    print("\nControls:")
    print("  - Wind Direction: Change wind direction (0-360°)")
    print("  - Wind Speed: Change wind intensity (0-1)")
    print("  - Base Probability: Base fire spread probability")
    print("  - Base Humidity: Average humidity level")
    print("\nButtons:")
    print("  - Step: Advance simulation by 1 timestep")
    print("  - Run/Pause: Start/stop continuous simulation")
    print("  - Reset: Reset simulation to initial state")
    print("  - Water Drop: Add water at center (increases humidity)")
    print("\nVisualization:")
    print("  - Main view: Current forest state")
    print("  - Humidity: Spatial humidity distribution")
    print("  - Elevation: Topography (fire spreads faster uphill)")
    print("  - Wind: Wind direction and speed")
    print("  - Statistics: Real-time metrics")
    print("  - Evolution: Time series of cell states")
    print("  - Burned area: Cumulative damage over time")
    print("\n" + "="*60)
    print("\nInitializing dashboard...")
    
    # Create and show dashboard
    dashboard = PhysicsDashboard(size=80)
    
    print("Dashboard ready! Close the window to exit.")
    dashboard.show()


if __name__ == "__main__":
    main()
