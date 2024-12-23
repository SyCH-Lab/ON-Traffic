import numpy as np
np.random.seed(12345)
import godunov as g
from pyDOE import lhs
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotProbeFD(t_train, x_train, rho_train, v_train):
      fig = plt.figure('Fundamental Diagram', figsize=(7.5, 5), facecolor='none', edgecolor='none', dpi=300)
      fig.patch.set_alpha(0)
      ax = fig.add_subplot(111, facecolor='none')
      ax.patch.set_alpha(0)
      for (t,x,u,v) in zip(t_train, x_train, rho_train, v_train):
            ax.scatter(u, v, c='k', s=5)
        
      plt.xlabel(r'$\rho$ [veh/km]')
      plt.ylabel(r'$v$ [km/min]')
    #   plt.tight_layout()
      plt.grid()
      plt.title('Fundamental Diagram')
      plt.show()


def plotDensityAtTimePoints(rho):
      fig, ax = plt.subplots(1, 3, figsize=(15, 5))
      for i, t in enumerate([0, 210, 374]):
            ax[i].plot(rho[:, t])
            ax[i].set_title(f'Time: {t}')
            plt.tight_layout()
      plt.show()


def plotCombinedDensityMovie(t_train, x_train, rho_train, v_train, rho):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Placeholder for the heatmap
        data = rho[:, 0][:, np.newaxis].T
        heatmap = ax.imshow(data, extent=[0, 500-1, 0, np.max(rho)], cmap='rainbow', vmin=0.0, vmax=1, aspect='auto')
        
        # Initialize an empty line plot for signal overlay
        line, = ax.plot([], [], lw=2, color='white')  # Using white for visibility against the heatmap
        

        ax.set_yticks([])  # Remove ticks on y-axis

        ax.set_xlabel('Position [m]')

        plt.tight_layout()
        
        # Placeholder for time annotation, positioned to be visible against the heatmap
        time_text = ax.text(0.87, 0.85, '', transform=ax.transAxes, color='black')
        
        # Initialize probe lines for each probe vehicle, making them visible from the start
        probe_lines = [ax.axvline(x[0], lw=3, color='black', visible=False) for x in x_train]  # Cyan for visibility
        probe_circles = [ax.scatter(x[0], rho_train[0][0], color='black', visible=False) for x in x_train]  # Red for visibility

        # Initialization function for the animation
        def init():
            heatmap.set_data(rho[:, 0][:, np.newaxis])
            line.set_data([], [])
            for pline, pcircle in zip(probe_lines, probe_circles):
                pline.set_visible(False)
                pcircle.set_visible(False)
            return [heatmap, line] + probe_lines + probe_circles

        # Update function for each frame
        def update(frame):
            # Updating the heatmap
            current_data = rho[:, frame]
            if current_data.ndim == 1:
                current_data = current_data[:, np.newaxis]
            heatmap.set_data(current_data.T)
            
            # Updating the line plot for the current frame
            line.set_data(range(500), rho[:, frame])
            
            # Updating time annotation
            time_text.set_text(f'Time: {frame/185.5:.2f} min')  # Assuming frame represents seconds, convert to minutes
            
            # Update probe lines based on current frame, checking against probe times
            for pv in range(len(probe_lines)):
                # probe_lines[pv].set_visible(False)  # Initially hide each line, only show if corresponding time matches
                for i, t in enumerate(t_train[pv]):
                    if int(t*185.5) == frame:  # Assuming t is in minutes, convert to frames
                        x_value = x_train[pv][i]*100
                        y_value = rho_train[pv][i]
                       
                        # probe_circles[pv].set_offsets([[x_value, y_value]])
                        probe_lines[pv].set_data([x_value, x_value], [0, y_value])
                        # probe_circles[pv].set_visible(True)
                        probe_lines[pv].set_visible(True)
                        
            
            return [heatmap, line, time_text] + probe_lines + probe_circles
        

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=375, init_func=init, blit=True)

        # Save the animation
        ani.save('combined_animation_godunov.gif', writer='pillow', fps=30, dpi=300, savefig_kwargs={'transparent': True})


def plotProbes(t_train, x_train, rho_train, v_train):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 

        # Plot Probe Density
        for (t, x, u) in zip(t_train, x_train, rho_train):
            sc1 = ax1.scatter(t, x, c=u, cmap='rainbow', vmin=0.0, vmax=1, s=5)
        ax1.set_xlabel(r'Time [min]')
        ax1.set_ylabel(r'Position [km]')
      #   ax1.set_ylim(0, self.L)
      #   ax1.set_xlim(0, self.Tmax)
        ax1.set_title('Probe Density')
        fig.colorbar(sc1, ax=ax1, label='Density')

        # Plot Probe Speed
        for (t, x, v) in zip(t_train, x_train, v_train):
            sc2 = ax2.scatter(t, x, c=v, cmap='rainbow', s=5)
        ax2.set_xlabel(r'Time [min]')
        ax2.set_ylabel(r'Position [km]')
      #   ax2.set_ylim(0, self.L)
      #   ax2.set_xlim(0, self.Tmax)
        ax2.set_title('Probe Speed')
        fig.colorbar(sc2, ax=ax2, label='Speed')

        plt.tight_layout()
        plt.savefig('probes_density_velocity.png', bbox_inches='tight')
        plt.show()

def plotProbeDensity(t_train, x_train, rho_train, v_train):

    fig = plt.figure(figsize = (10,10), facecolor='none', edgecolor='none', dpi=300)
    # Set the transparency of the figure
    fig.patch.set_alpha(0)

    # Create axes with transparent background
    ax = fig.add_subplot(111, facecolor='none')
    # Set the transparency of the axes
    ax.patch.set_alpha(0)
    for (t, x, u) in zip(t_train, x_train, rho_train):
        sc1 = ax.scatter(t, x, c=u, cmap='rainbow', vmin=0.0, vmax=1, s=5)
    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'Position [km]')
    ax.set_title('Density Measurements')
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 2)

    # Show the plot
    plt.show()


def plotDensityMovie(rho, t_train, x_train, rho_train, v_train):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], lw=2)  # Initialize an empty line plot
      #   ax.set_xlim(0, 249)  # Assuming x-axis represents the position, adjust as needed
        ax.set_ylim(0, np.max(rho))  # Adjust y-axis limits based on your data range
        ax.set_xlabel('Position [km]')
        ax.set_ylabel('Amplitude')
        time_text = ax.text(0.83, 0.85, '', transform=ax.transAxes)  # Placeholder for time annotation

        probe_lines = [ax.axvline(x[0]*100, lw=1, color='red', visible=False) for x in x_train] 

        # Initialization function for the animation
        def init():
            line.set_data([], [])
            for pline in probe_lines:
                pline.set_visible(False)  # Initially set all probe lines to be invisible
            return [line] + probe_lines

        # Update function for each frame
        def update(frame):
            line.set_data(range(500), rho[:, frame])  # Update the line plot for the current frame
            time_text.set_text(f'Time: {frame:.2f} s')  # Update time annotation
            for pv in range(len(probe_lines)):
                for i, t in enumerate(t_train[pv]):
                    if (int(t*187.5) == frame):
                        probe_lines[pv].set_data([x_train[pv][i]*100, x_train[pv][i]*100], [0,rho_train[pv][i]])
                        probe_lines[pv].set_visible(True)                    
                
            return [line, time_text]# + probe_lines

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=375, init_func=init, blit=True)

        # Save the animation
        ani.save('lineplot_animation_godunov.gif', writer='pillow', fps=50, dpi=50)


# rho [Nx, Nt], [500, 375]
# breakpoint()

# plotProbeFD(t_train, x_train, rho_train, v_train)
# plotDensityAtTimePoints(rho)
# plotProbeDensity(t_train, x_train, rho_train, v_train)
# plotCombinedDensityMovie(t_train, x_train, rho_train, v_train, rho)
# breakpoint()
# plotDensityMovie(rho, t_train, x_train, rho_train, v_train)

def plot_sensor_points(sensor_points):
    plt.scatter(sensor_points[:,1], sensor_points[:,0], c=sensor_points[:,2], cmap='viridis')
    plt.xlim([0,1])
    plt.xlabel('time')
    plt.ylabel('position')
    plt.grid()
    plt.title('sensor_points')
    plt.colorbar()
    plt.show()


def pad_to_shape(arr, target_shape, cap_shape=None):
    if arr.shape[0] < target_shape:
        arr = np.pad(arr, ((0, target_shape - arr.shape[0]), (0, 0)), mode='constant', constant_values=-2)
    if cap_shape is not None:
        arr = arr[-cap_shape:]
    return arr

# split first probe and boundary, then pad probe 
# def pad_to_shape_branch(coords, values, target_shape_boundary, target_shape_probe):
#     # Separate boundary and probe data based on ID in coords
#     boundary_data_coords = coords[coords[:, 2] == -1]
#     probe_data_coords = coords[coords[:, 2] != -1]
    
#     boundary_data_values = values[coords[:, 2] == -1]
#     probe_data_values = values[coords[:, 2] != -1]
    
#     # Truncate or pad boundary data to target_shape_boundary
#     filtered_boundary_coords = boundary_data_coords[:target_shape_boundary]
#     filtered_boundary_values = boundary_data_values[:target_shape_boundary]
    
#     # Truncate or pad probe data to target_shape_probe
#     if probe_data_coords.shape[0] < target_shape_probe:
#         # Pad if probe data is smaller than target_shape_probe
#         pad_size = target_shape_probe - probe_data_coords.shape[0]
#         filtered_probe_coords = np.pad(probe_data_coords, ((0, pad_size), (0, 0)), mode='constant', constant_values=-2)
#         filtered_probe_values = np.pad(probe_data_values, ((0, pad_size), (0, 0)), mode='constant', constant_values=-2)
#     else:
#         # Truncate if probe data is larger than target_shape_probe
#         filtered_probe_coords = probe_data_coords[:target_shape_probe]
#         filtered_probe_values = probe_data_values[:target_shape_probe]
    
#     # Combine boundary and probe data back together
#     filtered_coords = np.vstack([filtered_boundary_coords, filtered_probe_coords])
#     filtered_values = np.vstack([filtered_boundary_values, filtered_probe_values])
    
#     return filtered_coords, filtered_values


def pad_to_shape_branch(coords, values, target_shape_boundary, target_shape_probe):
    # Separate boundary and probe data based on ID in coords
    boundary_data_coords = coords[coords[:, 2] == -1]
    probe_data_coords = coords[coords[:, 2] != -1]
    
    boundary_data_values = values[coords[:, 2] == -1]
    probe_data_values = values[coords[:, 2] != -1]
    
    # Truncate or pad boundary data to target_shape_boundary
    filtered_boundary_coords = boundary_data_coords[:target_shape_boundary]
    filtered_boundary_values = boundary_data_values[:target_shape_boundary]
    
    # Truncate or pad probe data to target_shape_probe
    if probe_data_coords.shape[0] < target_shape_probe:
        # Pad if probe data is smaller than target_shape_probe
        pad_size = target_shape_probe - probe_data_coords.shape[0]
        
        # Generate random values from existing probe data
        random_indices = np.random.choice(probe_data_coords.shape[0], size=pad_size)
        random_coords = probe_data_coords[random_indices]
        random_values = probe_data_values[random_indices]
        
        # Concatenate original and random padded data
        filtered_probe_coords = np.concatenate([probe_data_coords, random_coords], axis=0)
        filtered_probe_values = np.concatenate([probe_data_values, random_values], axis=0)
    else:
        # Truncate if probe data is larger than target_shape_probe
        filtered_probe_coords = probe_data_coords[:target_shape_probe]
        filtered_probe_values = probe_data_values[:target_shape_probe]

    # Combine boundary and probe data back together
    filtered_coords = np.vstack([filtered_boundary_coords, filtered_probe_coords])
    filtered_values = np.vstack([filtered_boundary_values, filtered_probe_values])

    return filtered_coords, filtered_values


def memory_stats_npz(data):
    total_memory_usage = 0

    for key, array in data.items():
        if isinstance(array, np.ndarray):  # Only calculate memory for NumPy arrays
            memory_usage_bytes = array.nbytes  # Get the memory usage in bytes
            memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert to MB
            total_memory_usage += memory_usage_mb
            print(f"Array '{key}' uses {memory_usage_mb:.2f} MB")
        else:
            print(f"Key '{key}' is not an ndarray (likely an integer or other type).")

    # Print the total memory usage
    print(f"Total memory usage: {total_memory_usage:.2f} MB")
