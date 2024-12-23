import numpy as np
np.random.seed(42)
from godunov_vis_tools import * 
import os
from tqdm import tqdm



num_simulations = 100
save_dir = f"datasets/godunov_{num_simulations}"

Tmax = 25
L = 5
P = 7_000

# Multi-processing execution
if __name__ == "__main__":
   
    # Initialize lists to collect all the loaded data
    branch_coords_list, branch_values_list, output_sensor_coords_list, output_sensor_values_list, rho_list, v_list, x_list, t_list = [], [], [], [], [], [], [], []
    
    # Initialize placeholders for Nx and Nt
    Nx, Nt = None, None

    result_paths = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, file))]

    # Load all saved simulations
    for path in tqdm(result_paths):
        data = np.load(path)
        branch_coords_list.append(data['branch_coords'].astype(np.float16))
        branch_values_list.append(data['branch_values'].astype(np.float16))
        output_sensor_coords_list.append(data['output_sensor_coords'].astype(np.float16))
        output_sensor_values_list.append(data['output_sensor_values'].astype(np.float16))
        rho_list.append(data['rho'].astype(np.float16))
        # v_list.append(data['v'])
        x_list.append(data['x'].astype(np.float16))
        t_list.append(data['t'].astype(np.float16))

        # Set Nx and Nt based on the first simulation 
        if Nx is None and Nt is None:
            Nx = data['Nx']
            Nt = data['Nt']
     
    
    # Pad arrays and stack them
    print("start padding")
    max_shape = np.array([arr.shape for arr in branch_coords_list]).max()
    branch_coords_padded = np.array([pad_to_shape(arr, max_shape) for arr in branch_coords_list])
    branch_values_padded = np.array([pad_to_shape(arr, max_shape) for arr in branch_values_list])

    print("start stacking")
    branch_coords = np.stack(branch_coords_padded, axis=0)
    branch_values = np.stack(branch_values_padded, axis=0)
    output_sensor_coords = np.stack(output_sensor_coords_list, axis=0)
    output_sensor_values = np.stack(output_sensor_values_list, axis=0)
    rho = np.stack(rho_list, axis=0)
    # v = np.stack(v_list, axis=0)
    x = np.stack(x_list, axis=0)
    t = np.stack(t_list, axis=0)

    print("start saving")
    # Save all arrays, including global parameters
    combined_save_path = os.path.join(save_dir, f"../godunov_{num_simulations}_combined.npz")
    np.savez(combined_save_path, 
             branch_coords=branch_coords.astype(np.float16), 
             branch_values=branch_values.astype(np.float16), 
             output_sensor_coords=output_sensor_coords.astype(np.float16), 
             output_sensor_values=output_sensor_values.astype(np.float16), 
             rho=rho.astype(np.float16), x=x.astype(np.float16), t=t.astype(np.float16),
             Nx=Nx, Nt=Nt, 
             Xmax=L, Tmax=Tmax, P=P, N=num_simulations)
    