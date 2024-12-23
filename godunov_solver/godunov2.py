"""
Based on code from:
https://github.com/mBarreau/TrafficReconstructionIdentification
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from pyDOE import lhs

import torch

def flux(Vf, greenshield=True):
       
    if greenshield:
        rhoc = 0.4
        def f(rho):
            return Vf*rho*(1-rho)
    else: 
        rhoc = 0.3 
        def f(rho):
            return Vf*rho*(rho <= rhoc) + Vf*rhoc*(rho - 1)/(rhoc - 1)*(rho > rhoc)           
    return (f, rhoc)

class PhysicsSim:
    
    def __init__(self, L, Nx, Tmax, Vf=1, gamma=0.05, greenshield=True, key=42):
        self.Nx = Nx
        self.L = L
        self.Tmax = Tmax
        self.update(Vf, gamma)
        self.greenshield = greenshield
        
    def update(self, Vf, gamma):
        self.Vf = Vf
        self.gamma = gamma
        self.deltaX = self.L/self.Nx
        if gamma > 0:
            self.deltaT = 0.8*min(self.deltaX/Vf, self.deltaX**2/(2*gamma))
        else:
            self.deltaT = 0.8*self.deltaX/Vf
        self.Nt = int(np.ceil(self.Tmax/self.deltaT))
       
        
class ProbeVehicles:
    def __init__(self, sim, xiPos, xiT):
        self.sim = sim
        self.Nxi = len(xiPos)
        self.xi = [np.array([xiPos[i] * sim.Nx / sim.L], dtype=int) for i in range(self.Nxi)]
        self.xiT = [np.array([xiT[i] * sim.Nt / sim.Tmax], dtype=int) for i in range(self.Nxi)]
        self.xiArray = [np.array([xiPos[i]]) for i in range(self.Nxi)]
        self.xiTArray = [np.array([xiT[i]]) for i in range(self.Nxi)]

    def update(self, z, n):
        for j in range(self.Nxi):  # ODE for the agents
            if n * self.sim.Tmax / self.sim.Nt < self.xiTArray[j][-1]:
                continue
            
            # Update position using speed and deltaT
            new_pos = self.xiArray[j][-1] + self.sim.deltaT * self.speed(z[self.xi[j][-1]])
            
            # Apply periodic boundary conditions
            new_pos = new_pos % self.sim.L
            
            # Update arrays
            self.xiArray[j] = np.append(self.xiArray[j], new_pos)
            self.xiTArray[j] = np.append(self.xiTArray[j], n * self.sim.Tmax / self.sim.Nt)
            
            # Convert to indices
            self.xi[j] = np.append(self.xi[j], int(new_pos * self.sim.Nx / self.sim.L))
            self.xiT[j] = np.append(self.xiT[j], n)
            
    def speed(self, z):
        if z > 0:
            f,_ = flux(self.sim.Vf, greenshield=self.sim.greenshield)
            return f(z)/z
        else:
            return self.sim.Vf
    
    def getMeasurements(self, z):
        xMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        tMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        zMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        vMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        for j in range(self.Nxi):
            tMeasurements[j] = self.xiTArray[j][0:-1]
            xMeasurements[j] = self.xiArray[j][0:-1]
            for n in self.xiT[j][0:-1]:
                newDensity = z[self.xi[j][n],self.xiT[j][n]]
                zMeasurements[j] = np.append(zMeasurements[j], newDensity)
                vMeasurements[j] = np.append(vMeasurements[j], self.speed(newDensity))
                    
        return (xMeasurements, tMeasurements, zMeasurements, vMeasurements)
    
    def plot(self):
        for j in range(self.Nxi):
            plt.scatter(self.xiTArray[j], self.xiArray[j], c='k', alpha=0.5)
        

class BoundaryConditions:
    
    def __init__(self, sim, minZ0, maxZ0, rhoBar=-1, rhoSigma=0, sinePuls=15, key=42, length_scale=0.2):
        self.minZ0 = minZ0
        self.maxZ0 = maxZ0
        self.sinePuls = sinePuls
        self.sim = sim
        self.L = sim.L
        Tx = 0.4
        Tt = 0.20
        np.random.seed(key)
        torch.manual_seed(key)
        self.length_scale = length_scale
        self.Nx = sim.Nx
        self.Nt = sim.Nt
        
        
        self.X = truncnorm((minZ0 - rhoBar) / rhoSigma, 
                            (maxZ0 - rhoBar) / rhoSigma, 
                            loc=rhoBar, scale=rhoSigma)
        self.Npoints = [int(np.ceil(sim.Tmax/Tt)), int(np.ceil(sim.L/Tx))]
    
    def getBoundaryControl(self):
        def simulate_step_wavelet_boundary_condition(n, u_min, u_max, min_width=1, max_width=2, min_distance=1, max_distance=2):
            # Initialize the sample path with a constant value plus small random noise
            c = 0.3  # Mean value of the path (set to 0 in this case)
            u_b = np.random.normal(loc=c, scale=0.005, size=n)  # Small noise for the path
            
            # Track the position of the last wavelet
            last_wavelet_end = 0  # Start from the beginning of the array
            
            while last_wavelet_end < n:
                # Sample the length of the wavelet and downtime
                wavelet_length = np.random.uniform(min_width, max_width)
                downtime_length = np.random.uniform(min_distance, max_distance)
                
                # Calculate the ramp-up time as 20% of the wavelet length
                ramp_up_time = int(0.001 * wavelet_length)
                
                # Determine the start and end indices for the wavelet
                i = int(last_wavelet_end)
                j = min(int(i + wavelet_length), n)  # Ensure the wavelet doesn't exceed the array size
                
                # Calculate the end of the ramp-up phase
                ramp_end = min(i + ramp_up_time, j)
                
                # Apply a linear ramp-up from u_min to u_max
                if ramp_end > i:  # Check if there is a valid ramp-up phase
                    u_b[i:ramp_end] = np.linspace(u_min, u_max, ramp_end - i)
                
                # Assign the maximum value for the remainder of the wavelet range
                if ramp_end < j:  # Check if there is a remaining constant phase
                    u_b[ramp_end:j] = u_max
                
                # Update the position of the last wavelet end
                last_wavelet_end = j + downtime_length  # Add downtime after the wavelet
                
                # If the next wavelet would exceed the array size, break the loop
                if last_wavelet_end >= n:
                    break

            return u_b


        # Parameters
        n = self.Nt  # Discretization size in the t-dimension
        Tmax = self.sim.Tmax  # Maximum time
        u_min = 0.3  # Minimum solution bound
        u_max = 1  # Maximum solution bound

        # Generate wavelet boundary condition
        self.u_b = simulate_step_wavelet_boundary_condition(n, u_min, u_max, min_width=1*n/Tmax, max_width=2*n/Tmax, min_distance=1*n/Tmax, max_distance=2*n/Tmax)

        return self.u_b
    
    
# 
    
    def getZ0(self):
  
        def simulate_step_initial_condition(m, u_min, u_max, min_width, max_width):
            # Initialize the sample path starting from 0
            u_0 = np.zeros(m)

            # Sample the width of the first zero segment
            first_piece_width = np.random.randint(1*self.sim.Nx/self.L, 2*self.sim.Nx/self.L)

            # Set the first segment to the specified height (e.g., 0.1)
            u_0[:first_piece_width] = 0.1

            # Ensure the first segment has zero height
            i = first_piece_width  # The first `first_zero_width` points will be zero

            # Continue adding steps until no more space is available
            while i < m:
                # Calculate the remaining space
                remaining_space = m - i

                # If the remaining space is less than max_width, set the final step width to the remaining space
                if remaining_space <= max_width:
                    step_width = remaining_space
                else:
                    # Otherwise, sample the step width between min_width and max_width
                    step_width = np.random.randint(min_width, max_width + 1)

                # Determine the end index for this step
                j = min(m, i + step_width)

                # Sample a new step height without restrictions
                step_height = np.random.uniform(u_min, u_max)

                # Update the sample path with the new step height
                u_0[i:j] = step_height

                # Update the index for the next step
                i = j

            return u_0
        

        x = np.linspace(0, self.sim.L, self.sim.Nx)
        z0 = np.ones_like(x)*0

        m = self.sim.Nx  # discretization size in the x-dimension
        u_min = 0.1  # lower bound
        u_max = 1.0  # upper bound

        z0 = simulate_step_initial_condition(m, u_min, u_max, min_width=1*m/self.L, max_width=2*m/self.L)

        return z0
    
    def generate_random_gaussians(self, num_gaussians, mean_range, amplitude_range, width_range):
        gaussians = []
        # print(f"Generating {num_gaussians} random gaussians")
        for _ in range(num_gaussians):
            mean = np.random.uniform(*mean_range)
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            gaussians.append((mean, amplitude, width))
        return gaussians
    

    def getZbottom(self):
        
        points = self.sim.Tmax*lhs(1, samples=self.Npoints[0])
        points = (points/self.sim.deltaT).astype(np.int32)
        points = np.sort(points.reshape((self.Npoints[0],)))
        points = np.append(points, self.sim.Nt)
        zinValues = self.X.rvs((self.Npoints[0]+1, ))
        zin = np.ones((points[0], 1))*zinValues[0]
        for i in range(self.Npoints[0]):
            zin = np.vstack((zin, np.ones((points[i+1] - points[i], 1))*zinValues[i+1]))
       
            
        return zin
    
    def getZtop(self):
        
        points = self.sim.Tmax*lhs(1, samples=self.Npoints[0])
        points = (points/self.sim.deltaT).astype(np.int32)
        points = np.sort(points.reshape((self.Npoints[0],)))
        points = np.append(points, self.sim.Nt)
        zinValues = self.X.rvs((self.Npoints[0]+1, ))
        zinValues = np.ones((self.Npoints[0]+1, ))
        zin = np.ones((points[0], 1))*zinValues[0]
        for i in range(self.Npoints[0]):
            zin = np.vstack((zin, np.ones((points[i+1] - points[i], 1))*zinValues[i+1]))
      
        return zin
    
class SimuGodunov:

    def __init__(self, Vf, gamma, xiPos, xiT, zMin=0, zMax=1, L=5, Tmax=2, Nx = 300, rhoBar=-1, rhoSigma=0, greenshield=True, key=42, length_scale=0.2):
        
        self.sim = PhysicsSim(L, Nx, Tmax, Vf, gamma, greenshield, key=key)
        
        bc = BoundaryConditions(self.sim, zMin, zMax, rhoBar, rhoSigma, key=key, length_scale=length_scale)

        self.L = L
        self.length_scale = length_scale

        self.z0 = bc.getZ0()
        self.b0 = bc.getBoundaryControl()
        self.zBottom = bc.getZbottom()
        self.zTop = bc.getZtop()
        self.zBottom = np.ones_like(self.zBottom) * 0 + self.z0[-1]
        self.zTop = np.ones_like(self.zTop) * 0 + self.z0[0]
        
        
        self.pv = ProbeVehicles(self.sim, xiPos, xiT)
        
        self.zMax = zMax
        self.zMin = zMin
        
        
    def g(self, u, v):
    
        f, rhoc = flux(self.sim.Vf, greenshield=self.sim.greenshield)
        
        if u < 0: 
            u = 0;
        elif u >= 1:
            u = 1;
            
        if v < 0: 
            v = 0;
        elif v >= 1:
            v = 1;
        
        if u > v:
            if v >= rhoc:
                retour = f(v)
            elif u <= rhoc:
                retour = f(u)
            else:
                retour = f(rhoc)
        else:
            retour = min(f(u), f(v))
            
        return retour
    
    def simulation(self, use_periodic_bc=False):
        
        Nx = self.sim.Nx
        Nt = self.sim.Nt
        deltaX = self.sim.deltaX
        deltaT = self.sim.deltaT
        Vf = self.sim.Vf
        gamma = self.sim.gamma
        
        z = np.zeros((Nx, Nt))
        
        for i in range(Nx):
            z[i, 0] = self.z0[i]

        # Boundary conditions toggle
        for n in range(1, Nt):  # Apply numerical estimation
                        
            for i in range(0, Nx):  # Real traffic state, handle all points
                
                if use_periodic_bc:
                    # Periodic boundary conditions
                    if i == 0:  # First point (linked to last)
                        left = z[Nx-1, n-1]
                        right = z[i+1, n-1]
                    elif i == Nx-1:  # Last point (linked to first)
                        left = z[i-1, n-1]
                        right = z[0, n-1]
                    else:  # Interior points
                        left = z[i-1, n-1]
                        right = z[i+1, n-1]
                else:
                    # Custom boundary conditions
                    if i == 0:  # Left boundary (free)
                        left = z[i, n-1]  # Can be left as it is, "free" boundary
                        right = z[i+1, n-1]  # Normal right neighbor
                    elif i == Nx-1:  # Right boundary (periodically 0 or 1)
                        left = z[i-1, n-1]
                        # Right boundary value alternates between 0 and 0.95 based on time step
                        right = self.b0[n]
                    else:  # Interior points
                        left = z[i-1, n-1]
                        right = z[i+1, n-1]
                
                if gamma > 0:  # Heat equation
                    z[i, n] = z[i, n-1] + deltaT*(gamma*(left -
                            2*z[i, n-1] + right)/deltaX**2 -
                            Vf*(1-2*z[i, n-1])*(right - left)/(2*deltaX))
                else:  # Godunov scheme
                    gpdemi = self.g(z[i, n-1], right)
                    gmdemi = self.g(left, z[i, n-1])
                    z[i, n] = z[i, n-1] - deltaT*(gpdemi - gmdemi)/deltaX
                
                # Keep values within min/max bounds
                z[i, n] = min(max(z[i, n], self.zMin), self.zMax)

            self.pv.update(z[:, n], n)

        f, rhoc = flux(self.sim.Vf, greenshield=self.sim.greenshield)
        v = f(z)/z
        self.z = z
        return z, v



    def getAxisPlot(self):
        return (self.x, self.t)
        
    def plot(self):
        
        z = self.z           
        self.t = np.linspace(0, self.sim.Tmax, self.sim.Nt)
        self.x = np.linspace(0, self.sim.L, self.sim.Nx)
        
        # # fig = plt.figure(figsize=(7.5, 5))
        fig = plt.figure(figsize=(10, 10), facecolor='none', edgecolor='none', dpi=100)
        fig.patch.set_alpha(0)
        X, Y = np.meshgrid(self.t, self.x)
        plt.pcolor(X, Y, z, vmin=0.0, vmax=1.0, shading='auto', cmap='rainbow', rasterized=True)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.xlim(0, self.sim.Tmax)
        plt.ylim(0, self.sim.L)
        plt.gcf().patch.set_edgecolor('none')
        plt.colorbar()

        self.pv.plot()
        plt.legend(["Probe Trajectories"])
        plt.savefig('trajectories.png', transparent=True)

        fig = plt.figure(figsize=(10, 10), facecolor='none', edgecolor='none', dpi=100)
        plt.title('Density evolution over time')
        
        time_colors = np.linspace(0.8, 0.2, z.shape[1])  # Opacity decreases over time
        for t in reversed(range(1, z.shape[1], 20)):
            if t == 1:
                color = 'red'
                linewidth = 5  # Make the red line thicker
            else:
                color = 'black'
                linewidth = 1.0  # Default line width for other lines
            plt.plot(range(z.shape[0]), z[:, t], alpha=time_colors[t], label=f'Time {t+1}' if t in [0, z.shape[1]-1] else '', color=color, linewidth=linewidth)

        plt.ylim(0, 1)    
        plt.show()
        
    def getMeasurements(self, selectedPacket=-1, totalPacket=-1, noise=False):
        '''
        Collect data from N probe vehicles
    
        Parameters
        ----------
        selectedPacket : float64, optional
            Number of measurements per packet selected. If -1 then all 
            the measurements are used. 
            If a real number between [0, 1], this is the fraction of used 
            measurements.
            Otherwise, it is the number of measurements used within a packet. 
            It must be an integer less than totalPacket. 
            The default is -1.
        totalPacket : integer, optional
            Length of a packet. If -1, there is only one packet.
            The default is -1.
        noise : boolean, optional
            If True, noise is added on the measurements. The default is False.
    
        Returns
        -------
        x_selected : list of N numpy array of shape (?,1)
            space coordinate of the measurements.
        t_selected : list of N numpy array of shape (?,1)
            time coordinate of the measurements.
        rho_selected : list of N numpy array of shape (?,1)
            density measurements.
        v_selected : list of N numpy array of shape (?,1)
            velocity measurements.
    
        '''
        x_true, t, rho_true, v_true = self.pv.getMeasurements(self.z)
        Nxi = len(x_true)
     
        x_selected = []
        t_selected = []
        rho_selected = []
        v_selected = []
        for k in range(Nxi):
         
            Nt = t[k].shape[0]
         
            if totalPacket == -1:
                totalPacket = Nt
            if selectedPacket <= 0:
                selectedPacket = totalPacket
            elif selectedPacket < 1:
                selectedPacket = int(np.ceil(totalPacket*selectedPacket))
         
            nPackets = int(np.ceil(Nt/totalPacket))
            toBeSelected = np.empty((0,1), dtype=np.int32)
            for i in range(nPackets):
                randomPackets = np.arange(i*totalPacket, min((i+1)*totalPacket, Nt), dtype=np.int32)
                np.random.shuffle(randomPackets)
                if selectedPacket > randomPackets.shape[0]:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:-1])
                else:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:selectedPacket])
            toBeSelected = np.sort(toBeSelected) 
            
            if noise:
                noise_trajectory = np.random.normal(0, 1.5, Nt)/1000
                noise_trajectory = np.cumsum(noise_trajectory.reshape(-1,), axis=0)
                noise_meas = np.random.normal(0, 0.02, Nt).reshape(-1,)
            else:
                noise_trajectory = np.array([0]*Nt)
                noise_meas = np.array([0]*Nt)
                
            x_selected.append(np.reshape(x_true[k][toBeSelected] + noise_trajectory[toBeSelected], (-1,1)))
            rho_temp = rho_true[k][toBeSelected] + noise_meas[toBeSelected]
            rho_selected.append(np.reshape(np.maximum(np.minimum(rho_temp, 1), 0), (-1,1)))
            t_selected.append(np.reshape(t[k][toBeSelected], (-1,1)))
            v_selected.append(np.reshape(v_true[k][toBeSelected], (-1,1)))
    
        return t_selected, x_selected, rho_selected, v_selected
    
    def getDatas(self, x, t):
        X = (x/self.sim.deltaX).astype(int)
        T = (t/self.sim.deltaT).astype(int)
        return self.z[X, T]
    
    def getPrediction(self, tf, Nexp=10, wMax=30, Amax=1, Amin=0): 
        Nplus = int((tf-self.sim.Tmax)/self.sim.deltaT) 
        wRand = wMax*np.random.rand(Nexp, 2) 
        Arand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2) 
        Brand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2) 
        Crand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2) 
         
        t = np.linspace(self.sim.Tmax, tf, Nplus) 
         
        boundaryValues = np.zeros((Nplus, Nexp*2)) 
        for i in range(Nexp): 
            boundaryValues[:,2*i] = Crand[i,0] + Arand[i,0]*np.sin(wRand[i,0]*t) + Brand[i,0]*np.cos(wRand[i,0]*t) 
            boundaryValues[:,2*i+1] = Crand[i,1] + Arand[i,1]*np.sin(wRand[i,1]*t) + Brand[i,1]*np.cos(wRand[i,1]*t) 
        boundaryValues = np.maximum(boundaryValues, 0) 
        boundaryValues = np.minimum(boundaryValues, 1) 
         
        return (t, boundaryValues)


def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)