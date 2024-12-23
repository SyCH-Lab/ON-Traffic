import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils



def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def solve_ADR(Nx=100, Nt=100, P=10_000, length_scale=0.2, m=100, 
              key=0, xmax=1, tmax=1, k_coef=0.01, g_coef=0.01):
    """Solve 1D ADR equation without JAX dependencies."""
    xmin, xmax = 0, xmax
    tmin, tmax = 0, tmax
    k = lambda x: k_coef * np.ones_like(x) 
    v = lambda x: 0 * np.ones_like(x)
    g = lambda u: g_coef * u ** 2
    dg = lambda u: 2 * g_coef * u

    # Generate random numbers
    torch.manual_seed(key)  # Use a fixed seed for reproducibility
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = np.dot(L, torch.randn(N).numpy())
    u0_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Grid setup
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    k = k(x)
    v = v(x)

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * np.eye(Nx - 2) + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(v[2:] - v[: Nx - 2])
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * np.eye(Nx - 2) - M[1:-1, 1:-1] - v_bond

    # Initial condition
    u = np.zeros((Nx, Nt))
    u[:, 0] = u0_fn(x)

    # Boundary conditions
    u[0, :] = 0  # left boundary
    u[-1, :] = 0  # right boundary

    # Time-stepping
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * gi
        b2 = (c - h2dgi) @ u[1:-1, i]
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)

    # Output
    xx = np.linspace(xmin, xmax, m)
    u0 = u0_fn(xx)

    # set the boundary of ic to zero !!!
    u0[0] = 0
    u0[-1] = 0

    # idx = torch.randint(0, max(Nx, Nt), (P, 2))
    idx_x = torch.randint(0, Nx, (P,))  # Indices for x, size Nx
    idx_t = torch.randint(0, Nt, (P,))  # Indices for t, size Nt

    # use the correct indices
    y = np.concatenate([x[idx_x][:, None], t[idx_t][:, None]], axis=1)
    s = u[idx_x, idx_t]
    return (x, t, u), (u0, y, s)


def gen_dataset_scatter_trunk(N=500, Nx=100, Nt=100, P=10_000, length_scale=0.2, m=100, 
                xmax=1, tmax=1, k_coef=0.01, g_coef=0.01, random_seed=0):
    
    torch.manual_seed(random_seed)
    
    keys = torch.randint(-2**31, 2**31, (N,))
    x = []
    t = []
    u = []
    u0 = []
    y = []
    s = []

    for key in keys:
        (x_, t_, u_), (u0_, y_, s_) = solve_ADR(Nx=Nx, Nt=Nt, P=P, length_scale=length_scale, m=m, 
                                                  key=key.item(), xmax=xmax, tmax=tmax, k_coef=k_coef, g_coef=g_coef)
        x.append(x_)
        t.append(t_)
        u.append(u_)
        u0.append(u0_)
        y.append(y_)
        s.append(s_)

    x = torch.tensor(np.array(x), dtype=torch.float32)
    t = torch.tensor(np.array(t), dtype=torch.float32)
    u = torch.tensor(np.array(u), dtype=torch.float32)
    u0 = torch.tensor(np.array(u0), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    s = torch.tensor(np.array(s), dtype=torch.float32)

    return x, t, u, u0, y, s



def gen_dataset_lagragian(N=500, Nx=100, Nt=100, P=10_000, length_scale=0.2, m=100, 
                xmax=1, tmax=1, k_coef=0.01, g_coef=0.01, random_seed=0):
    
    torch.manual_seed(random_seed)
    
    keys = torch.randint(-2**31, 2**31, (N,))
    x = []
    t = []
    u = []
    u0 = []
    y = []
    s = []

    history_timesteps = 10
    spatial_skips = 10

    # define boolean index grid
    grid = np.zeros((100, 100))
    positions = np.linspace(0, 100, 11, dtype=int)

    for ti in range(10):
        for pos in positions:
            # Use modulo to wrap around the indices
            grid[(pos + ti) % 100, ti] = 1
    
    grid = grid

    for key in keys:
        (x_, t_, u_), (u0_, y_, s_) = solve_ADR(Nx=Nx, Nt=Nt, P=P, length_scale=length_scale, m=m, 
                                                  key=key.item(), xmax=xmax, tmax=tmax, k_coef=k_coef, g_coef=g_coef)
        x.append(x_)
        t.append(t_)
        u.append(u_)
        
        X, T = np.meshgrid(x_, t_)
        coords = [X[:, :, None], T[:, :, None]]
        coords = np.concatenate(coords, axis=-1)
        values_coords_all = np.concatenate([u_[None, :].T, coords], axis=-1)
        values_coords = []

        for i in range(100):
            for j in range(100):
                if grid[i, j] == 1:
                    values_coords.append(values_coords_all[j, i])


        u0.append(values_coords)
        y.append(y_)
        s.append(s_)

    x = torch.tensor(np.array(x), dtype=torch.float32)
    t = torch.tensor(np.array(t), dtype=torch.float32)
    u = torch.tensor(np.array(u), dtype=torch.float32)
    u0 = torch.tensor(np.array(u0), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    s = torch.tensor(np.array(s), dtype=torch.float32)

    return x, t, u, u0, y, s


def gen_dataset_multi_branch(N=500, Nx=100, Nt=100, P=10_000, length_scale=0.2, m=100, 
                xmax=1, tmax=1, k_coef=0.01, g_coef=0.01, random_seed=0):
    
    '''
    x_train: torch.Size([N, Nx]), t_train: torch.Size([N, Nt]), UU_train: torch.Size([N, Nx, Nt])
    u_train: torch.Size([N, Nx*num_timesteps_input, 3]), y_train: torch.Size([N, P, 2]), s_train: torch.Size([N, P])
    '''
    
    torch.manual_seed(random_seed)
    
    keys = torch.randint(-2**31, 2**31, (N,))
    x = []
    t = []
    u = []
    u0 = []
    y = []
    s = []

    num_timesteps_input = 3

    for key in keys:
        (x_, t_, u_), (u0_, y_, s_) = solve_ADR(Nx=Nx, Nt=Nt, P=P, length_scale=length_scale, m=m, 
                                                  key=key.item(), xmax=xmax, tmax=tmax, k_coef=k_coef, g_coef=g_coef)
        x.append(x_)
        t.append(t_)
        u.append(u_)
        coords = np.array(np.meshgrid(x_, t_[0:num_timesteps_input], indexing='ij')).reshape(2, -1).T
        u0.append(np.concatenate([u_[:, 0:num_timesteps_input].flatten()[:, None], coords], axis=1))
        y.append(y_)
        s.append(s_)

    x = torch.tensor(np.array(x), dtype=torch.float32)
    t = torch.tensor(np.array(t), dtype=torch.float32)
    u = torch.tensor(np.array(u), dtype=torch.float32)
    u0 = torch.tensor(np.array(u0), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    s = torch.tensor(np.array(s), dtype=torch.float32)

    return x, t, u, u0, y, s




def gen_dataset_grid_trunk(N=500, Nx=100, Nt=100, P=10_000, length_scale=0.2, m=100, 
                xmax=1, tmax=1, k_coef=0.01, g_coef=0.01, random_seed=0):
    
    torch.manual_seed(random_seed)
    
    keys = torch.randint(-2**31, 2**31, (N,))
    x = []
    t = []
    u = []
    u0 = []
    y = []
    s = []

    for key in keys:
        (x_, t_, u_), (u0_, y_, s_) = solve_ADR(Nx=Nx, Nt=Nt, P=P, length_scale=length_scale, m=m, 
                                                  key=key.item(), xmax=xmax, tmax=tmax, k_coef=k_coef, g_coef=g_coef)
        x.append(x_)
        t.append(t_)
        u.append(u_)
        u0.append(u0_)
        y.append(np.array(np.meshgrid(x_, t_, indexing='ij')).reshape(2, -1).T)
        s.append(u_.reshape(-1))

    x = torch.tensor(np.array(x), dtype=torch.float32)
    t = torch.tensor(np.array(t), dtype=torch.float32)
    u = torch.tensor(np.array(u), dtype=torch.float32)
    u0 = torch.tensor(np.array(u0), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    s = torch.tensor(np.array(s), dtype=torch.float32)

    return x, t, u, u0, y, s



def plot_data(x, t, UU, u, y, s):
    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    # plt.imshow(UU.T, origin='lower', cmap='jet')
    plt.scatter(y[:,0]*100, y[:,1]*100, c=s, s=10, cmap='jet')
    plt.colorbar()
    plt.subplot(1,2,2)
    time_colors = np.linspace(1, 0.2, UU.shape[1])  # Opacity decreases over time

    for t in reversed(range(0, UU.shape[1], 3)):
        plt.plot(range(UU.shape[0]), UU[:, t], alpha=time_colors[t], label=f'Time {t+1}' if t in [0, UU.shape[1]-1] else '')
    plt.grid()
    plt.xlim([0, UU.shape[0]])

    plt.tight_layout()


def plot_data_results_evolution(UU, prediction):
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2,3,1)  # First plot in the first row
    time_colors = np.linspace(1, 0.2, UU.shape[1])  # Opacity decreases over time
    for t in reversed(range(0, UU.shape[1], 3)):
        plt.plot(range(UU.shape[0]), UU[:, t], alpha=time_colors[t], label=f'Time {t+1}' if t in [0, UU.shape[1]-1] else '')
    plt.grid()
    plt.title("Ground Truth")
    plt.xlim([0, UU.shape[0]])

    plt.subplot(2,3,2)  # Second plot in the first row
    time_colors = np.linspace(1, 0.2, prediction.shape[1])  # Opacity decreases over time
    for t in reversed(range(0, prediction.shape[1], 3)):
        plt.plot(range(prediction.shape[0]), prediction[:, t], alpha=time_colors[t], label=f'Time {t+1}' if t in [0, prediction.shape[1]-1] else '')
    plt.grid()
    plt.title("Prediction")
    plt.xlim([0, prediction.shape[0]])

    plt.subplot(2,3,3)  # Third plot in the first row
    time_colors = np.linspace(1, 0.2, prediction.shape[1])  # Opacity decreases over time
    for t in reversed(range(0, prediction.shape[1], 3)):
        plt.plot(range(prediction.shape[0]), np.abs(prediction[:, t]-UU[:, t]), alpha=time_colors[t], label=f'Time {t+1}' if t in [0, prediction.shape[1]-1] else '')
    plt.grid()
    plt.title("Error")
    plt.xlim([0, prediction.shape[0]])

    plt.subplot(2,3,4)  # First plot in the second row
    # error at each time step
    plt.plot(range(UU.shape[1]), np.abs(prediction - UU).mean(axis=0))
    plt.grid()
    plt.title("Mean Absolute Error over time")
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error')

    plt.subplot(2,3,5)  # Second plot in the second row
    # mean absolute error over space
    plt.plot(range(UU.shape[0]), np.abs(prediction - UU).mean(axis=1))
    plt.grid()
    plt.title("Mean Absolute Error over space")
    plt.xlabel('Space')
    plt.ylabel('Mean Absolute Error')

    # plt.subplot(2,3,6)  # Third plot in the second row

    plt.tight_layout()
    plt.show()

    plt.tight_layout()



class DeepONetDataset(Dataset):
    def __init__(self, branch_coord, branch_values, trunk_coords, targets, UU, device, t_start=None):

        self.branch_coord = branch_coord.to(device)  
        if len(self.branch_coord.shape) == 2:
            self.branch_coord = self.branch_coord.unsqueeze(-1)
           
        # self.branch_values = branch_values.unsqueeze(-1).to(device)
        self.branch_values = branch_values.to(device)
        self.trunk_coords = trunk_coords.to(device)
        self.targets = targets.to(device)
        self.UU = UU.to(device)
        if t_start is not None:
            self.t_start = t_start.to(device)
        else:
            self.t_start = None
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.t_start is not None:
            return self.branch_coord[idx], self.branch_values[idx], self.trunk_coords[idx], self.targets[idx], self.UU[idx], self.t_start[idx]
        else:
            return self.branch_coord[idx], self.branch_values[idx], self.trunk_coords[idx], self.targets[idx], self.UU[idx]
    

class DeepONetDatasetTrain(Dataset):
    def __init__(self, branch_coord, branch_values, trunk_coords, targets, device):

        self.branch_coord = branch_coord#.to(device)  
        if len(self.branch_coord.shape) == 2:
            self.branch_coord = self.branch_coord.unsqueeze(-1)
           
        self.branch_values = branch_values
        self.trunk_coords = trunk_coords
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.branch_coord[idx], self.branch_values[idx], self.trunk_coords[idx], self.targets[idx]


def filter_branch_horizon(branch_coords, branch_values, t_past=1, t_pred=10, t_start=None, max_shift=4/15, shift_to_zero=True):
    if t_start is None:
        t_start = np.random.uniform(0, max_shift)
        # t_start = np.random.choice([0, 5])
    t = t_start + t_past

    filtered_branch_coords = []
    filtered_branch_values = []

    max_shape = 0

    for i in range(len(branch_coords)):
        mask_boundary = branch_coords[i][:, 2] == -1
        mask_probes = branch_coords[i][:, 2] > -1

        filter_boundary = torch.logical_and(branch_coords[i][:, 1] >= t_start, branch_coords[i][:, 1] <= t + t_pred)
        filter_probes = torch.logical_and(branch_coords[i][:, 1] >= t_start, branch_coords[i][:, 1] <= t)

        filtered_boundary_coords = branch_coords[i][torch.logical_and(mask_boundary, filter_boundary)]
        filtered_boundary_values = branch_values[i][torch.logical_and(mask_boundary, filter_boundary)]

        filtered_probes_coords = branch_coords[i][torch.logical_and(mask_probes, filter_probes)]
        filtered_probes_values = branch_values[i][torch.logical_and(mask_probes, filter_probes)]

        # Concatenate boundary and probe filtered results
        final_filtered_coords = torch.cat([filtered_boundary_coords, filtered_probes_coords], dim=0)
        final_filtered_values = torch.cat([filtered_boundary_values, filtered_probes_values], dim=0)

        # Shift the time horizon such that t_start = 0
        if shift_to_zero:
            final_filtered_coords[:, 1] -= t_start

        filtered_branch_coords.append(final_filtered_coords)
        filtered_branch_values.append(final_filtered_values)

        max_shape = max(max_shape, final_filtered_coords.shape[0])
        # print(f"Branch {i}: {final_filtered_coords.shape}")

    # print(f"Max shape: {max_shape}")
    # Pad the lists of variable-sized tensors and stack them
    filtered_branch_coords_padded = rnn_utils.pad_sequence(filtered_branch_coords, batch_first=True, padding_value=-2)
    filtered_branch_values_padded = rnn_utils.pad_sequence(filtered_branch_values, batch_first=True, padding_value=-2)

    return filtered_branch_coords_padded[:,:max_shape,:], filtered_branch_values_padded[:,:max_shape,:], t_start


def filter_trunk_horizon(trunk_coords, targets, t_past=1, t_pred=10, t_start=None):

    t = t_start + t_past

    min_shape = 7_000

    filtered_trunk_coords_list = []
    filtered_targets_list = []

    for i in range(len(trunk_coords)):
        mask = torch.logical_and(trunk_coords[i][:, 1] >= t_start, trunk_coords[i][:, 1] <= t + t_pred)
        filtered_trunk_coords = trunk_coords[i][mask]
        filtered_targets = targets[i][mask]

        # Shift the time horizon such that t_start = 0
        filtered_trunk_coords[:, 1] -= t_start

        filtered_trunk_coords_list.append(filtered_trunk_coords)
        filtered_targets_list.append(filtered_targets)

        min_shape = min(min_shape, filtered_trunk_coords.shape[0])
        # print(f"Trunk {i}: {filtered_trunk_coords.shape}") 

    # cut shapes to min_shape and stack them ( no padding)
    filtered_trunk_coords_padded = torch.stack([filtered_trunk_coords[:min_shape] for filtered_trunk_coords in filtered_trunk_coords_list])
    filtered_targets_padded = torch.stack([filtered_targets[:min_shape] for filtered_targets in filtered_targets_list])

    # print(f"Min shape: {min_shape}")
    return filtered_trunk_coords_padded, filtered_targets_padded


def sample_branch_inputs_keep_boundary(branch_coords, branch_values, min_to_keep=0.67):
    # Ensure the tensors are on the same device
    device = branch_coords.device
    
    # Get the number of samples in the batch and the number of coordinates per sample
    batch_size, m, _ = branch_coords.shape[:3]  # Assuming last dimension contains (x, t, ID)
    
    # Extract the IDs (third column in branch_coords)
    ids = branch_coords[:, :, 2]
    
    # Find the minimum number of ID == -1 per batch entry and sample half of it
    min_IDs = (ids == -1).sum(dim=1).min().item()
    sampled_min_IDs = min_IDs // 2  # Use only half of the min number of ID == -1
    sampled_min_IDs = min_IDs  # Use only half of the min number of ID == -1
    
    # Find the maximum number of valid samples (where ID != -1 and ID != -2)
    valid_mask = (ids != -1) & (ids != -2)
    min_valid_samples = valid_mask.sum(dim=1).min().item()  # Maximum number of valid IDs available across the batch

    # Select a fixed number of valid samples for all entries
    m_sampled_valid = torch.randint(int(min_valid_samples * min_to_keep), min_valid_samples + 1, (1,), device=device).item()

    # Initialize lists to store sampled coordinates and values
    sampled_branch_coords = []
    sampled_branch_values = []
    
    for i in range(batch_size):
        # IDs for the current sample
        current_ids = ids[i]
        
        # Get indices for ID == -1 and keep exactly half of `min_IDs`
        id_neg1_indices = (current_ids == -1).nonzero(as_tuple=False).squeeze()
        
        # Randomly select half of the minimum ID == -1 samples
        sampled_neg1_indices = id_neg1_indices[torch.randperm(len(id_neg1_indices), device=device)[:sampled_min_IDs]]
        
        # Get remaining valid indices (where ID != -1 and ID != -2)
        valid_indices = (current_ids != -1) & (current_ids != -2)
        valid_indices = valid_indices.nonzero(as_tuple=False).squeeze()

        # Randomly select `m_sampled_valid` from valid indices
        sampled_valid_indices = valid_indices[torch.randperm(len(valid_indices), device=device)[:m_sampled_valid]]
      
        # Combine the sampled ID == -1 indices and the randomly sampled valid indices
        final_indices = torch.cat([sampled_neg1_indices, sampled_valid_indices])

        # Sample the branch coordinates and values using the final indices
        sampled_branch_coords.append(branch_coords[i, final_indices])
        sampled_branch_values.append(branch_values[i, final_indices])

    # Convert the lists of sampled coordinates and values back to tensors
    sampled_branch_coords = torch.stack(sampled_branch_coords)
    sampled_branch_values = torch.stack(sampled_branch_values)

    return sampled_branch_coords, sampled_branch_values



def sample_branch_inputs(branch_coords, branch_values):
    # Ensure the tensors are on the same device
    device = branch_coords.device
    
    # Get the number of samples in the batch and the number of coordinates per sample
    batch_size, m = branch_coords.shape[:2]
    
    # Initialize lists to store sampled coordinates and values
    sampled_branch_coords = []
    sampled_branch_values = []

    # Select a random number of samples (m_sampled) for this batch entry
    m_sampled = torch.randint(m // 3, m + 1, (1,)).item()
    
    for i in range(batch_size):
       
        # Randomly select m_sampled indices
        sampled_indices = torch.randperm(m, device=device)[:m_sampled]
        
        # Sample the branch coordinates and values using the same indices
        sampled_branch_coords.append(branch_coords[i, sampled_indices])
        sampled_branch_values.append(branch_values[i, sampled_indices])
    
    # Convert the lists of sampled coordinates and values back to tensors
    sampled_branch_coords = torch.stack(sampled_branch_coords)
    sampled_branch_values = torch.stack(sampled_branch_values)
    
    return sampled_branch_coords, sampled_branch_values


def sample_trunk_inputs(trunk_coords, targets, m_sampled=None):
    # Ensure the tensors are on the same device
    device = trunk_coords.device
    
    # Get the number of samples in the batch and the number of coordinates per sample
    batch_size, m = trunk_coords.shape[:2]
    
    # Initialize lists to store sampled coordinates and targets
    sampled_trunk_coords = []
    sampled_targets = []

    if m_sampled is None:
        m_sampled = torch.randint(m//5, m + 1, (1,)).item()
    
    for i in range(batch_size):
        
        # Randomly select m_sampled indices
        sampled_indices = torch.randperm(m, device=device)[:m_sampled]
        
        # Sample the trunk coordinates and targets using the same indices
        sampled_trunk_coords.append(trunk_coords[i, sampled_indices])
        sampled_targets.append(targets[i, sampled_indices])
    
    # Convert the lists of sampled coordinates and targets back to tensors
    sampled_trunk_coords = torch.stack(sampled_trunk_coords)
    sampled_targets = torch.stack(sampled_targets)
    
    return sampled_trunk_coords, sampled_targets


def calculate_memory_usage(tensors, dtype_size=4):
    """
    Calculate memory usage of given tensors.
    
    Args:
    tensors (list): List of tensors or their shapes.
    dtype_size (int): Size of each element in bytes (default is 4 for float32).
    
    Returns:
    float: Total memory usage in MB.
    """
    total_memory = 0
    for tensor in tensors:
        # Calculate the number of elements in the tensor
        num_elements = torch.tensor(tensor.shape).prod().item()
        memory = num_elements * dtype_size  # Memory in bytes
        total_memory += memory
    
    # Convert to MB
    total_memory_mb = total_memory / (1024 ** 2)
    return total_memory_mb