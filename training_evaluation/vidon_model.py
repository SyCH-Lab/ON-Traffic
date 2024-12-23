import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # self.activation = SineActivation()
        self.activation = activation

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            # Glorot/Xavier initialization adapted for tanh or sigmoid      # TODO: maybe change initialization for ReLU
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            # nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    
class Head(nn.Module):
    def __init__(self, d_head, d_enc, p):
        super(Head, self).__init__()
        self.d_enc = torch.tensor(d_enc)
        self.d_head = torch.tensor(d_head)
        self.head_value = MLP([d_enc, d_head, d_head, d_head, d_head, p])
        self.head_weight = MLP([d_enc, d_head, d_head, d_head, d_head, 1])
        
    def forward(self, x):
        # x: (batch_size, m, d_enc)
        assert len(x.shape) == 3, f"Expected 3D input, got {x.shape}, check batching"
        weight_tilde = self.head_weight(x) # (batch_size, m, 1)
        weight = torch.softmax(weight_tilde / torch.sqrt(self.d_enc), dim=1) # (batch_size, m, 1)
        value_tilde = self.head_value(x)    # (batch_size, m, p) 
        value = torch.mean(weight * value_tilde, dim=1)  # batch_size, p
        return value

class Branch(nn.Module):
    def __init__(self, d=2, d_v=1, d_enc=40, num_heads=4, d_head=128, d_combiner=256, p=50, UQ=False):
        super(Branch, self).__init__()
        self.coord_encoder = MLP([d, d_enc, d_enc, d_enc, d_enc])
        self.value_encoder = MLP([d_v, d_enc, d_enc, d_enc, d_enc])
        self.heads = nn.ModuleList([Head(d_head, d_enc, p) for _ in range(num_heads)])
        self.UQ = UQ
    
        self.combiner = MLP([num_heads*p, d_combiner, d_combiner, d_combiner, d_combiner, p], activation=nn.ReLU())
        if self.UQ:
            self.combiner_sigm = MLP([num_heads*p, d_combiner//2, d_combiner//2, d_combiner//2, p//2], activation=nn.ReLU())
        
    def forward(self, coord, value):
        # filter out inputs that are -2 # TODO: check if this is the right way to handle missing data
        
        coord_enc = self.coord_encoder(coord)   # (m, d_enc)
        value_enc = self.value_encoder(value)   # (m, d_enc)
        encoding = coord_enc + value_enc        # (m, d_enc)
        head_outputs  = [head(encoding) for head in self.heads] # [(p,), (p,), (p,), (p,)]
        combined = self.combiner(torch.cat(head_outputs, dim=-1)) # (p,)

        if self.UQ:
            combined_sigm = self.combiner_sigm(torch.cat(head_outputs, dim=-1))
            return combined, combined_sigm

        return combined
    
class Trunk(nn.Module):
    def __init__(self, d=2, d_trunk=250, p=50, UQ=False):
        super(Trunk, self).__init__()
        self.UQ=UQ
        if self.UQ:
            self.shared_trunk = MLP([d, d_trunk, d_trunk, d_trunk], activation=nn.ReLU())
            self.mean_trunk = MLP([d_trunk, d_trunk, d_trunk, p], activation=nn.ReLU())
            self.sigm_trunk = MLP([d_trunk, d_trunk//4, d_trunk//4, p//2], activation=nn.ReLU())
        else:
            self.trunk = MLP([d, d_trunk, d_trunk, d_trunk, d_trunk, d_trunk, p], activation=nn.ReLU())
        
    def forward(self, x):
        if self.UQ:
            x = self.shared_trunk(x)
            mean = self.mean_trunk(x)
            sigm = self.sigm_trunk(x)
            return mean, sigm

        return self.trunk(x)


class NonlinearDecoder(nn.Module):
    def __init__(self, d=1, d_dec=50, p=50, activation=nn.ReLU()):  
        super(NonlinearDecoder, self).__init__()
        self.decoder = MLP([p, d_dec, d_dec, d_dec, d_dec, d], activation=activation)
        
    def forward(self, x):
        x = self.decoder(x)
        return x

class VIDON(nn.Module):
    def __init__(self, p=50, num_heads=4, d_branch_input=1, d_v=1, use_linear_decoder=False, UQ=False):
        super(VIDON, self).__init__()
        
        self.branch = Branch(d=d_branch_input, d_v=d_v, d_enc=50, num_heads=num_heads, d_head=128//2, d_combiner=256//3, p=p, UQ=UQ)
        self.trunk = Trunk(d=2, d_trunk=300, p=p, UQ=UQ)
        self.nonlinear_decoder = NonlinearDecoder(d=1, d_dec=p//2, p=p, activation=nn.ReLU())
        self.UQ = UQ
        if self.UQ:
            self.nonlinear_decoder_sigm = NonlinearDecoder(d=1, d_dec=p//2, p=p//2, activation=nn.ReLU())

        self.bias = nn.Parameter(torch.ones(1)*0)                # TODO: make sure this is also trainable
        self.use_linear_decoder = use_linear_decoder
        
        
    def forward(self, branch_coord, branch_value, trunk_input):
        # branch_coords: (batch_size, m, d)
        # branch_value: (batch_size, m, d_v)
        # trunk_input: (batch_size, P, 2)
        
        basisfunctions = self.trunk(trunk_input)                # (batch_size, P, p)     
        coefficients = self.branch(branch_coord, branch_value)  # (batch_size, p)

        if self.UQ:
            mean = coefficients[0].unsqueeze(1) * basisfunctions[0]
            sigm = coefficients[1].unsqueeze(1) * basisfunctions[1]
            mean_decoded = self.nonlinear_decoder(mean) + self.bias
            sigm_decoded = self.nonlinear_decoder_sigm(sigm)
            return mean_decoded.squeeze(), sigm_decoded.squeeze()
       
    
        latent = coefficients.unsqueeze(1) * basisfunctions # (Nt*Nx,)
        x = self.nonlinear_decoder(latent) + self.bias
        x = x.squeeze()
        return x
    

class SepTrunk(nn.Module):
    def __init__(self, d=2, d_trunk=250, p=50):
        super(SepTrunk, self).__init__()
        self.trunk_x = MLP([1, d_trunk, d_trunk, d_trunk, d_trunk, d_trunk, p], activation=nn.ReLU())
        self.trunk_t = MLP([1, d_trunk, d_trunk, d_trunk, d_trunk, d_trunk, p], activation=nn.ReLU())
        
    def forward(self, x):
        x_input, t_input = x[:,:,0:1], x[:,:,1:2]
        x_output = self.trunk_x(x_input)
        t_output = self.trunk_t(t_input)
        
        return x_output * t_output
    


class FDLearner(nn.Module):
    def __init__(self, d_FD=10):
        super(FDLearner, self).__init__()
        self.FD = MLP([1, d_FD, d_FD, d_FD, d_FD, 1], activation=nn.Sigmoid())
        
    def forward(self, x):
        x = self.FD(x)
        return x
    