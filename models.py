
import matplotlib.pyplot as plt

import torch; print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

import torchvision
import pandas as pd 
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from IPython.core.display import Image, display

import numpy as np
import shutil; import random

from functools import wraps

import matplotlib
import matplotlib.pyplot as plt

import cv2
from obstacle_tower_env import ObstacleTowerEnv


device = "cuda" if torch.cuda.is_available() else "cpu"; print("device", device)

CONTROLLER_MODEL_PATH = "controller.torch"

# VAE
SZ=128
nz = 300
h_dim = 16384 #8192#16384 #32768; # this results from img sz and model architecture
ENCODER_MODEL_PATH = 'encoder_v2.torch'
GEN_MODEL_PATH = 'gen_v2.torch'

Z_SEQS_PATH = "z_seq_sequential.pt"

# LSTM
N_IN = nz 
N_HIDDEN = 1024; 
N_LAYERS = 2; 
N_GAUSSIANS = 5
DROP = .1
LSTM_MODEL_PATH = 'lstm.torch'



N_ACTIONS = 11; # just panning

s = nn.Softmax()
class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(nz+N_HIDDEN, N_ACTIONS)
        self.act = nn.Sigmoid()
        #self.act = nn.LeakyReLU(0.2, inplace=True)
        #self.s = nn.LogSoftmax()
        #self.s = nn.Softmax()
    
    def forward(self, x):
        x = self.act(self.linear(x))
        #x = self.linear(x)
        # softmax each action set
        # front = s(x[:, :3]); pan=s(x[:, 3:6]); jump=s(x[:, 6:8]); side=s(x[:, 8:11])
        # cat back together
        #x = torch.cat([front,pan,jump,side],dim=1)
        return x #front, pan, jump, side
    

def action_probs_to_action(probs):
    forward = probs[:, 0:3]; camera=probs[:, 3:6]; jump=probs[:,6:8]; side=probs[:, 8:11]
    action = [torch.distributions.Categorical(p).sample().detach().item() for p in [forward,camera,jump,side]]
    return action

def action_to_dummies(action):
    return [
    1. if action[0]==0 else 0., # no frontal
    1. if action[0]==1 else 0., # forward
    1. if action[0]==2 else 0., # backward
    1. if action[1]==0 else 0., # camera none
    1. if action[1]==1 else 0., # camera left
    1. if action[1]==2 else 0., # camera right
    1. if action[2]==0 else 0., # no jump
    1. if action[2]==1 else 0., # jump
    1. if action[3]==0 else 0., # no side
    1. if action[3]==1 else 0., # side right
    1. if action[3]==2 else 0., # side left
    ]

    
    
    
# These come from zseqs straight from vae, we train lstm using these values to normalize.
Z_MEAN_PATH = 'z_mean.torch'
Z_MAX_PATH = 'z_max.torch'

class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, bn=True, pad=None):
        super().__init__()
        if pad is None: pad = ks//2//stride
        self.conv = nn.Conv2d(ni, no, ks, stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(no) if bn else None
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x
    
class Encoder(nn.Module):
    # image size, number of channels in image, number of channels out in next layer, number of extra layers
    def __init__(self, isize, nc, ndf):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.initial = ConvBlock(nc, ndf, 4, 2, bn=False)
        csize,cndf = isize/2,ndf

        pyr_layers = []
        while csize > 4:
        #while csize > 8:
            pyr_layers.append(ConvBlock(cndf, cndf*2, 4, 2))
            cndf *= 2; csize /= 2
        self.pyramid = nn.Sequential(*pyr_layers)
        
        self.fc1 = nn.Linear(h_dim, nz)
        #self.bn1 = nn.BatchNorm1d(z_dim)
        
        self.fc2 = nn.Linear(h_dim, nz)
        #self.bn2 = nn.BatchNorm1d(z_dim)

    def forward(self, input):
        x = self.initial(input)
        x = self.pyramid(x)
        global h
        h = x.view(x.size(0), -1) # flatten 
        mu, logvar = self.fc1(h), self.fc2(h)
        logvar = torch.clamp(logvar, -6, 6)
        
        #mu, logvar = self.bn1(h1), self.bn2(h2)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    
    
class DeconvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, pad, bn=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ni, no, ks, stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(no)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x

class Generator(nn.Module):
    def __init__(self, isize, nz, nc, ngf):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        self.drop_z = nn.Dropout(p=0.1)
        
        cngf, tisize = ngf//2, 4 # This is original
        #cngf, tisize = ngf//2, 8 # RG 9.13 reducing size of network
        while tisize!=isize: cngf*=2; tisize*=2
        layers = [DeconvBlock(nz, cngf, 4, 1, 0)]

        csize, cndf = 4, cngf
        while csize < isize//2:
            layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 1))
            cngf //= 2; csize *= 2

        layers.append(nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        self.features = nn.Sequential(*layers)
        
        self.reward_head = nn.Linear(nz, 1)

    def forward(self, z_seq): 
        reward = self.reward_head(z_seq).squeeze(1)
        
        z_seq = z_seq.unsqueeze(-1).unsqueeze(-1)
        z_seq = self.drop_z(z_seq)
        out = F.sigmoid(self.features(z_seq)) # changed this to Sigmoid to force to 0 to 1
        return out, reward

    
    
def triangle_lr(epoch, n_epochs):
    peak_epoch = 3; n_e = n_epochs; peak_lr = 1e-3; min_lr = 1e-4;
    if epoch <= peak_epoch: lr = (((peak_lr-min_lr)/peak_epoch)*epoch + min_lr)
    else: 
        slope = (min_lr-peak_lr)/(n_e-peak_epoch); y_intercept = min_lr - slope*n_e
        lr = (slope*epoch + y_intercept)
    lr = np.round(lr,5);
    return lr


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p:float=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p) # swapping axes bc fastai does batch first
        return x * m

class WeightDropout(nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module:nn.Module, weight_p:float, layer_names=['weight_hh_l0']):
        super().__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()

def one_param(m): 
    "Return the first parameter of `m`."
    return next(m.parameters())

class AWD_LSTM(nn.Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182. from fastai"
    initrange=0.1
    def __init__(self, emb_sz:int, n_hid:int, n_layers:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, weight_p:float=0.5):
        super().__init__()
        self.ndir = 2 if bidir else 1
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, n_hid,
            1, bidirectional=bidir) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        
        self.z_pi = nn.Linear(n_hid, N_GAUSSIANS*N_IN)
        self.z_sigma = nn.Linear(n_hid, N_GAUSSIANS*N_IN)
        self.z_mu = nn.Linear(n_hid, N_GAUSSIANS*N_IN)
        
        self.N_HIDDEN = N_HIDDEN; self.N_IN=N_IN
        
    def forward(self, input, hiddens):
        raw_output = self.input_dp(input)
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, hiddens[l])
            new_hidden.append(new_h); raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        hidden = new_hidden
        pi = self.z_pi(raw_output) 
        # Do i even need softmax here? Moved this into loss function itself
        #pi = F.softmax(pi, dim=2) #ONNX can't handle dim 2
        #pi = pi / TEMPERATURE
            
        sigma = self.z_sigma(raw_output)
        sigma = torch.exp(sigma)
        #sigma = sigma * (TEMPERATURE ** 0.5)
        
        mu = self.z_mu(raw_output)
        
        return [pi, mu, sigma], hidden, raw_output
    
    
def make_hidden(bsz):
    return [(torch.zeros(1, bsz, N_HIDDEN).to(device),
            torch.zeros(1, bsz, N_HIDDEN).to(device))
            for l in range(N_LAYERS)]  



# Could probably replace this w batchnorm layer at end of encoder
def normalize_sequential(raw_data, calc=True, mean=None, absmax=None):
    """
    Takes in 2-d data (len sequence, len latent) representing entire uninterupted sequence. Normalizes and standardizes 
    to -1 to 1 based on mean, max of EACH latent dim separately
    """
    if type(raw_data) != torch.Tensor: raw_data=torch.tensor(raw_data).to(device)
    z_mean = raw_data.mean(dim=0).unsqueeze(0) if calc else mean
    centered = raw_data - z_meandf
    m, _ = centered.max(dim=0); mm, _ = centered.min(dim=0); 
    m_abs = torch.max(torch.abs(m),torch.abs(mm)).unsqueeze(0) if calc else absmax;
    normed_data = centered / m_abs
    return normed_data, z_mean, m_abs

def denormalize_sequential(raw_data, data_mean, data_abs): return raw_data * data_abs + data_mean


# create new dataset by randomly inverting waves to create challenge only MDN can handle
def generate_flipped_sine(L,N):
    T = 5
    L = 10000 # to create X and y, need to pad by one
    N = 3 # when zipping together, lose one.

    noise = np.random.uniform(-0.1, 0.1, (N, L)).astype(np.float32)
    x = np.empty((N, L), 'float32')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64') #+ noise

    rand_data = []
    for ii in range(len(data)):
        seq = data[ii]
        dd = [seq[0]]
        invert = False
        for i in range(1, len(data[0])):
            d = seq[i]; d_prev = seq[i-1]
            if d < 0 and d_prev > 0 and np.random.random() > .50: invert = True
            elif d > 0 and d_prev < 0: invert = False
            if invert: dd.append(seq[i]*-1)
            else: dd.append(seq[i])
        rand_data.append(dd)

    seq = np.transpose(np.array(rand_data), (1,0)); print(seq.shape) 
    return seq


def compute_returns(rewards, dones, times):
    # Add time rewards to reward
    rewards = np.copy(rewards); dones=np.copy(dones); times=np.copy(times)
    
    # Time rewards DO add density to rewards. normal rewards not capturing orbs. 
    # ALso verified getting keys DOES add reward of .1
    time_reward = np.round(((times[1:] - times[:-1]) / 10000), 4)
    time_reward = np.insert(time_reward, 0, 0.0)
    time_reward = np.array([t if t > -.25 else 0. for t in time_reward]) # on reset, don't want big negative reward
    #rewards += time_reward

    # Compute discounted returns
    GAMMA = .95
    R = []
    rewards=list(rewards); dones=list(dones)
    rewards.reverse(); dones.reverse(); time_reward=np.flip(time_reward)
    r2=0
    R.append(r2)
    for i in range(1, len(rewards)):
        if dones[i] or (rewards[i+1]==1.0): r1=0. # Reset to zero when done, also when go to new floor
        else: 
            r = .5 if rewards[i]==1.0 else rewards[i] # Reducing importance of floor completion.
            r1 = r + time_reward[i] + r2 * GAMMA
        R.append(np.round(r1, 4))
        r2=r1
    rewards.reverse(); R.reverse(); time_reward=np.flip(time_reward)
    return pd.DataFrame({"state_reward":rewards, 
                         "time_reward": time_reward, 
                         "returns":R})


# helper functions to toggle btwn trainable and not trainable. Sets requires_grad on and off.

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))
    
def to_np(t):
    return t.detach().cpu().numpy()
