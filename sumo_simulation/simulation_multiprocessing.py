import os, sys, csv

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci # traffic control interface
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from probe_vehicle import ProbeVehicleArray
from sumolib import checkBinary
import random


def run_sumo_simulation(index):

    # set random seed for reproducibility also for SUMO and numpy
    random.seed(index)
    np.random.seed(index)

    scenario = "highway"
    sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", scenario+"/"+scenario+".sumocfg"]

    traci.start(sumoCmd)

    deltaX = 0.010 # in km, more than a vehicle
    L = 3
    deltaT = traci.simulation.getDeltaT()/60 # in min
    Tmax = 15 # in min
    Tstart = 0 # in min
    sigma = 0.01*2 # in km
    tau = 0.06*2 # in min

    Nt = int(np.ceil(Tmax/deltaT)) # Number of temporal points
    NtStart = int(np.floor(Tstart/deltaT)) # Start of the simulation
    Nx = int(np.ceil(L/deltaX)) # Number of spatial points

    numberOfVehicles = np.zeros((Nx, Nt-NtStart))
    PVList = ProbeVehicleArray()

    def generate_schedule(Nt, min_duration=60, max_duration=120):
        schedule = []
        current_time = 0

        # First quarter of the simulation time has a continuous green light
        first_quarter_time = 100
        schedule.append({
            "red_start": None,  # No red light in the first quarter
            "red_end": None,
            "green_start": current_time,
            "green_end": first_quarter_time
        })
        current_time = first_quarter_time

        # Generate regular red-green cycles for the rest of the simulation
        while current_time < Nt:
            red_duration = random.randint(min_duration, max_duration)
            green_duration = random.randint(min_duration, max_duration)
            cycle_time = red_duration + green_duration
            schedule.append({
                "red_start": current_time,
                "red_end": current_time + red_duration,
                "green_start": current_time + red_duration,
                "green_end": current_time + cycle_time
            })
            current_time += cycle_time

        return schedule

    # Define the number of time steps in the simulation
    schedule = generate_schedule(Nt)
    current_cycle_index = 0
    traffic_light_status = []  # List to track traffic light status over time


    # Steady inflow of vehicles during the simulation
    for n in range(Nt):

        # Check if we need to move to the next cycle
        if current_cycle_index < len(schedule):
            cycle = schedule[current_cycle_index]
            
            # If we've reached the end of the current cycle, move to the next one
            if n >= cycle["green_end"]:
                current_cycle_index += 1
                if current_cycle_index < len(schedule):
                    cycle = schedule[current_cycle_index]

            # Determine current light phase based on the explicit time points
            if cycle["red_start"] is not None and cycle["red_start"] <= n < cycle["red_end"]:
                traffic_light_status.append(0)  # Red light
                # Apply speed limit only to vehicles in the first 10 meters of gneE2
                for vehID in traci.vehicle.getIDList():
                    edgeID = traci.vehicle.getRoadID(vehID)
                    vehPos = traci.vehicle.getLanePosition(vehID)  # Position along the lane

                    if edgeID == "gneE2" and vehPos <= 50:  # First 10 meters
                        traci.vehicle.setSpeed(vehID, 0)  # Stop vehicles
                    elif edgeID == "gneE2":
                        traci.vehicle.setSpeed(vehID, 20.33)  # Allow normal speed for the rest
            elif cycle["green_start"] <= n < cycle["green_end"]:
                traffic_light_status.append(1)  # Green light
                # Green light: allow normal speed for all vehicles on gneE2
                for vehID in traci.vehicle.getIDList():
                    edgeID = traci.vehicle.getRoadID(vehID)
                    if edgeID == "gneE2":
                        traci.vehicle.setSpeed(vehID, 20.33)
                

        # Update vehicle information if n is within the relevant time range
        if n >= NtStart:
            for vehID in traci.vehicle.getIDList():
                vehPos = traci.vehicle.getPosition(vehID)[0]
                if vehPos >= L * 1000:
                    continue

                vehSpeed = traci.vehicle.getSpeed(vehID)
                i = int(np.floor(vehPos / (1000 * deltaX)))
                if 0 <= i < Nx:
                    numberOfVehicles[i, n - NtStart] += 1
                    if traci.vehicle.getTypeID(vehID) == 'PV':
                        PVList.update(vehID, n * deltaT, vehPos / 1000, vehSpeed * 60 / 1000)

        traci.simulationStep()

    traci.close()

    t = np.linspace(Tstart, Tmax, Nt-NtStart)
    x = np.linspace(0, L, Nx)
    X, Y = np.meshgrid(t, x)

    maxI = int(np.ceil(5*sigma/deltaX))
    maxJ = int(np.ceil(5*tau/deltaT))
    kernel = np.zeros((2*maxI+1, 2*maxJ+1))
    for i in range(2*maxI+1):
        for j in range(2*maxJ+1):
            newI = i-maxI-1
            newJ = j-maxJ-1
            kernel[i,j] = np.exp(-abs(newI)*deltaX/sigma - abs(newJ)*deltaT/tau)
    N = kernel.sum()
    density = signal.convolve2d(numberOfVehicles, kernel, boundary='symm', mode='same')/N
    # densityMax = np.percentile(density, 95)
    densityMax = np.max(density)
    density = density/densityMax
    density = np.clip(density, 0, 1)

    def load_csv(file):
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        data = np.array(data).astype(np.float)
        return data

    tVar = []
    tVarPlot = []
    xVar = []
    rhoPV = []
    vPV = []
    for pv in PVList.pvs:
        tVar = tVar + pv.getT((NtStart-1)/60)
        tVarPlot = tVarPlot + pv.t
        xVar = xVar + pv.x
        vPV = vPV + pv.v
        for k in range(len(pv.t)):
            j = int(pv.t[k]/deltaT)
            i = int(pv.x[k]/deltaX)
            rhoPV.append(density[i,j-NtStart])

    with open(scenario+f'/dataset_idm/spaciotemporal{index}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[L, (Tmax-Tstart)]])
        writer.writerows(density)
        
    with open(scenario+f'/dataset_idm/pv{index}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.array([xVar, tVar, rhoPV, vPV]).T)

    # Save traffic light status as .npz
    traffic_light_status_filename = f"{scenario}/dataset_idm/traffic_light_status_{index}.npz"
    np.savez(traffic_light_status_filename, traffic_light_status=traffic_light_status)

    return f"Simulation with index {index} completed."


import multiprocessing
from tqdm import tqdm

# Function to run multiple simulations in parallel
def run_multiple_simulations_in_parallel(num_simulations, num_cores):
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(run_sumo_simulation, range(0, num_simulations)), total=num_simulations))
    return results

# Example usage
if __name__ == "__main__":
    scenario = "highway"
    num_simulations = 30  # Total number of simulations to run
    num_cores = 20  # Number of CPU cores to use

    results = run_multiple_simulations_in_parallel(num_simulations, num_cores)

    for result in results:
        print(result)