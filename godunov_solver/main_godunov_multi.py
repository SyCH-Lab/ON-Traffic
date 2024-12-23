import numpy as np
np.random.seed(42)
import godunov2 as g2
from pyDOE import lhs
import matplotlib.pyplot as plt
from godunov_vis_tools import * 
from tqdm import tqdm
import multiprocessing as mp  # Multi-processing
import concurrent.futures  # For multi-threading/multi-processing
import imageio
import os

# General parameters
Vf = 1.5
gamma = 0
Tmax = 25
p = 0.5
L = 5
rhoBar = 0.2
rhoSigma = 0.45
rhoMax = 200
noise = False
greenshield = False
Ncar = rhoBar * rhoMax * L
Npv = int(Ncar * p * 2)
length_scale = 0.1
max_history = Tmax
skip_timestep = 5
include_boundary_sensors = True

num_simulations = 100
start_key=0
keys = np.arange(start_key, num_simulations+start_key)
first_key = keys[0]
P = 7_000

# Initialize lists to collect results
branch_coords_list, branch_values_list, output_sensor_coords_list, output_sensor_values_list, rho_list, x_list, t_list, v_list = [], [], [], [], [], [], [], []

save_dir = f"datasets/godunov_{num_simulations}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

lower_bound_pos = -1*L
upper_bound_pos = 1*L

# Function to run the simulation for each key
def run_simulation(i):    
    xiPos = lower_bound_pos + (upper_bound_pos - lower_bound_pos) * lhs(1, samples=Npv, criterion='maximin', iterations=5).reshape((Npv,))
    xiPos = np.flip(np.sort(xiPos))
    xiT = np.array([0] * Npv)

    simu_godunov = g2.SimuGodunov(Vf, gamma, xiPos, xiT, L=L, Tmax=Tmax,
                                  zMin=0, zMax=1, Nx=250, greenshield=greenshield,
                                  rhoBar=rhoBar, rhoSigma=rhoSigma, key=i, length_scale=length_scale)

    rho, v = simu_godunov.simulation()
    Nx = simu_godunov.sim.Nx
    Nt = simu_godunov.sim.Nt

    # plot the simulation
    # simu_godunov.plot()
    

    t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)

    sensor_points = []
    for pv in range(len(t_train)):
        for t in range(0, len(t_train[pv]), skip_timestep):
            if t_train[pv][t] <= max_history and 0 < x_train[pv][t] < L:
                sensor_point = [x_train[pv][t].item(), t_train[pv][t].item(), rho_train[pv][t].item(), v_train[pv][t].item(), pv]
                sensor_points.append(sensor_point)

    if include_boundary_sensors:
        sample_times = np.linspace(0, Tmax, Nt)
        for index, t in enumerate(sample_times):
            if index % skip_timestep == 0:
                sensors_point_left = [0, t, rho[0, index].item(), 0, -1]
                sensor_point_right = [L, t, rho[-1, index].item(), 0, -1]
                sensor_points.append(sensors_point_left)
                sensor_points.append(sensor_point_right)

    sensor_points = np.array(sensor_points)
    branch_coords = sensor_points[:, [0, 1, -1]]
    branch_values = sensor_points[:, [2, 3]]

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, Tmax, Nt)
    idx_x = np.random.randint(0, Nx, (P,))
    idx_t = np.random.randint(0, Nt, (P,))
    output_sensor_coords = np.concatenate([x[idx_x][:, None], t[idx_t][:, None]], axis=1)
    output_sensor_values = rho[idx_x, idx_t]


    # After generating all the required data
    simulation_data = {
        'branch_coords': branch_coords.astype(np.float16),
        'branch_values': branch_values.astype(np.float16),
        'output_sensor_coords': output_sensor_coords.astype(np.float16),
        'output_sensor_values': output_sensor_values.astype(np.float16),
        'rho': rho.astype(np.float16),
        # 'v': v.astype(np.float16),
        'x': x.astype(np.float16),
        't': t.astype(np.float16),
        'Nx': Nx,
        'Nt': Nt
    }

    # Save the result as a separate file for each simulation
    save_path = os.path.join(save_dir, f"simulation_{i}.npz")
    np.savez(save_path, **simulation_data)

    return save_path

num_cores = os.cpu_count()

# Multi-processing execution
if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        result_paths = list(tqdm(executor.map(run_simulation, keys), total=len(keys)))
