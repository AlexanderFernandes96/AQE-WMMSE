from copy import deepcopy
import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import torch
# from ray import tune
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.schedulers import ASHAScheduler
import pprint
import sys
from pathlib import Path

# Disable the loading bars:
DISABLE_TQDM = False
# DISABLE_TQDM = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)

def load_complex(dataset_dir, variable_name_real, variable_name_imag):
    return (np.loadtxt(dataset_dir + variable_name_real + ".csv", delimiter=',') +
            1j * np.loadtxt(dataset_dir + variable_name_imag + ".csv", delimiter=','))

    # Hau = (K, M)    AP  -> UE
    # Har = (N, M)    AP  -> RIS
    # Hru = (K, N)    RIS -> UE
    # W   = (M, K)    UE stream -> AP

class EncoderLayer(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, Nc_enc):
        super(EncoderLayer, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.N_RIS = N_RIS
        self.Nc_enc = Nc_enc
        ## height: Hout = floor((Hin + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        ## width:  Wout = floor((Win + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        p = 0.5 # dropout probability
        H = N_RIS + 2 * K_UE * M_AP
        self.linear_encoder = nn.Sequential(
            nn.Linear(2*(K_UE*M_AP + K_UE*N_RIS + N_RIS*M_AP + K_UE*M_AP) + N_RIS, 32*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(32*H),
            nn.Linear(32*H, 16*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(16*H),
            nn.Linear(16*H, 8*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(8*H),
            nn.Linear(8*H, 4*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(4*H),
            # nn.Linear(4*H, Nc_enc + 2*K_UE*M_AP), # include W
            nn.Linear(4*H, Nc_enc), # don't include W
        )

    def forward(self, x):
        theta = x[0].float()
        w_r   = torch.flatten(x[1], 1).float()
        w_i   = torch.flatten(x[2], 1).float()
        hra_r = torch.flatten(x[3], 1).float()
        hra_i = torch.flatten(x[4], 1).float()
        hur_r = torch.flatten(x[5], 1).float()
        hur_i = torch.flatten(x[6], 1).float()
        hua_r = torch.flatten(x[7], 1).float()
        hua_i = torch.flatten(x[8], 1).float()
        x_in = torch.cat((theta, w_r, w_i, hra_r, hra_i, hur_r, hur_i, hua_r, hua_i), 1)
        theta_enc = self.linear_encoder(x_in)
        return theta_enc

# [1] W. Xu, L. Gan, and C. Huang, “A Robust Deep Learning-Based Beamforming Design for RIS-Assisted Multiuser MISO
# Communications With Practical Constraints,” IEEE Trans. Cogn. Commun. Netw., vol. 8, no. 2, pp. 694–706, Jun. 2022,
# doi: 10.1109/TCCN.2021.3128605.
class DNN(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS):
        super(DNN, self).__init__()
        H = N_RIS + 2*K_UE*M_AP
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.N_RIS = N_RIS
        p = 0.5 # dropout probability
        self.linear_encoder = nn.Sequential(
            nn.Linear(2*(K_UE*M_AP + K_UE*N_RIS + N_RIS*M_AP), 32*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(32*H),
            nn.Linear(32*H, 16*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(16*H),
            nn.Linear(16*H, 8*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(8*H),
            nn.Linear(8*H, 4*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(4*H),
            nn.Linear(4*H, H),
        )
    def forward(self, x):
        hra_r = torch.flatten(x[3], 1).float()
        hra_i = torch.flatten(x[4], 1).float()
        hur_r = torch.flatten(x[5], 1).float()
        hur_i = torch.flatten(x[6], 1).float()
        hua_r = torch.flatten(x[7], 1).float()
        hua_i = torch.flatten(x[8], 1).float()
        x_in = torch.cat((hra_r, hra_i, hur_r, hur_i, hua_r, hua_i), 1)
        x_out = self.linear_encoder(x_in)
        theta = x_out[:, 0:self.N_RIS]
        Wr = x_out[:, self.N_RIS:self.N_RIS+(self.K_UE*self.M_AP)]
        Wi = x_out[:, self.N_RIS+(self.K_UE*self.M_AP):self.N_RIS+2*(self.K_UE*self.M_AP)]
        W = Wr + 1j*Wi
        return theta, W


class QuantizerLayer(nn.Module):
    def __init__(self, C_code_words, dev):
        super(QuantizerLayer, self).__init__()
        ## a = amplitude, b = shift, c = slope
        ## for i = 0 to C_code_words-1:
        ##   z += a[i] * tanh( c[i] * (theta_n - b[i]) )
        ## where theta_n is a scalar value: [-pi, +pi) and z is the quantized theta_n
        self.a = torch.nn.Parameter(
            data=torch.from_numpy(
                np.ones(C_code_words-1) * np.pi / ( C_code_words )
            ), requires_grad=False)
        spacing = np.linspace(-1, 1, C_code_words + 1) * np.pi
        self.b = torch.nn.Parameter(
            data=torch.from_numpy(
                np.delete(spacing,[0,-1])
                # np.random.rand(C_code_words - 1, 1) * 2 * np.pi - 3 * np.pi),
            ), requires_grad=True)
        c_slope = 0.5
        # c_slope = 10
        if len(self.b) > 1:
            self.c = torch.nn.Parameter(
                # data=torch.from_numpy((15 / np.mean(np.diff(self.b.data.numpy()))) * np.ones(C_code_words - 1)),
                data=torch.from_numpy((c_slope * 2 * np.pi / np.mean(np.diff(self.b.data.numpy())) ) * np.ones(C_code_words - 1)),
                # data=torch.from_numpy(0.9 * np.ones(C_code_words - 1)),
                requires_grad=True)
        else:
            self.b = torch.nn.Parameter(data=torch.from_numpy(np.zeros(C_code_words - 1)), requires_grad=True)
            self.c = torch.nn.Parameter(
                data=torch.from_numpy(c_slope * np.ones(C_code_words - 1)),
                # data=torch.from_numpy(15 / np.pi * np.ones(C_code_words - 1)),
                requires_grad=True)
        self.C_code_words = C_code_words
        self.hardQ = False
        self.device = dev

    def forward(self, theta_enc):
        ## theta_enc = matrix (number of samples of train/test dataset, Nc_enc), in the range: [-pi, +pi)
        theta_qnt = torch.zeros(theta_enc.shape[0], theta_enc.shape[1], device=self.device)
        for i in range(self.C_code_words - 1):
            if self.hardQ:
                theta_qnt += self.a[i] * torch.sign(self.c[i] * (theta_enc - self.b[i]))
            else:
                theta_qnt += self.a[i] * torch.tanh(self.c[i] * (theta_enc - self.b[i]))
        return theta_qnt

    def plot_vals(self):
        a = np.array(self.a.cpu().detach().numpy())
        b = np.array(self.b.cpu().detach().numpy())
        c = np.array(self.c.cpu().detach().numpy())
        print(a,b,c, flush=True)
        if len(b) > 1:
            bdiff = np.max(np.diff(b))
            x_vals = np.linspace(np.min(b) - bdiff, np.max(b) + bdiff, 10000)
        else:
            x_vals = np.linspace(np.min(b) - np.pi, np.max(b) + np.pi, 10000)
        soft_quant = []
        hard_quant = []
        for x_val in x_vals:
            soft_quant.append(np.sum(a * np.tanh(c * (x_val - b))))
            hard_quant.append(np.sum(a * np.sign(c * (x_val - b))))
        return x_vals, soft_quant, hard_quant

class DecoderLayer(nn.Module):
    def __init__(self, N_RIS, Nc_enc):
        super(DecoderLayer, self).__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(Nc_enc, N_RIS),
            nn.ReLU(),
            nn.Linear(N_RIS, N_RIS),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(N_RIS, N_RIS), # best to make output layer a linear operator
            # nn.LeakyReLU(), # LeakyReLU or ReLU will make negative phase shifts not work
            # nn.Tanh(), # Note Tanh at output makes training harder considering the optimal value is periodic wrt 2pi
        )

    def forward(self, theta_qnt):
        theta_dec = self.linear_decoder(theta_qnt)
        theta_out = self.out_layer(theta_dec)
        return theta_out


class WupdateLayer(nn.Module):
    def __init__(self, K_UE, M_AP):
        super(WupdateLayer, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        p = 0.5 # dropout probability
        self.linear_W = nn.Sequential(
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 2*K_UE*M_AP),
        )
        self.linear_UL = nn.Sequential(
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 4*K_UE*M_AP),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4*K_UE*M_AP, 3*K_UE),
        )

    # def forward(self, theta, W_r, W_i, x):
    def forward(self, theta, x):
        # Solving optimal wmmse parameters with deep learning inspired by:
        #  [1] W. Xia, G. Zheng, Y. Zhu, J. Zhang, J. Wang, and A. P. Petropulu, “A deep learning framework for
        #  optimization of MISO downlink beamforming,” IEEE Trans. Commun., vol. 68, no. 3, pp. 1866–1880, Mar. 2020,
        #  doi: 10.1109/TCOMM.2019.2960361.

        theta_c = torch.exp(1j * theta) # complex
        # W_in = W_r + 1j*W_i
        # W_in = torch.reshape(W_in, (-1, self.M_AP, self.K_UE))
        # normfactor = torch.linalg.matrix_norm(W_in, ord='fro')
        # W_in = (10 ** (trainparams['snr_dB'] / 10)) * W_in / normfactor[:, None, None] # normalize W
        # W_in_r = torch.flatten(torch.real(W_in), start_dim=1)
        # W_in_i = torch.flatten(torch.imag(W_in), start_dim=1)
        # W_in_r = W_r
        # W_in_i = W_i
        # theta_r = torch.real(theta_c)
        # theta_i = torch.imag(theta_c)
        # theta_opt = x[0].float()
        Wr_opt = torch.flatten(x[1].float(), start_dim=1)
        Wi_opt = torch.flatten(x[2].float(), start_dim=1)
        Har = x[3].float() + 1j*x[4].float()
        Hru = x[5].float() + 1j*x[6].float()
        Hau = x[7].float() + 1j*x[8].float()

        # The effective channel that includes the compressed phase-shifts used at the RIS
        H = Hau + torch.einsum("bkn, bnm -> bkm", Hru, torch.einsum("bn, bnm -> bnm", theta_c, Har))
        h_r = torch.flatten(torch.real(H), start_dim=1)
        h_i = torch.flatten(torch.imag(H), start_dim=1)

        # hra_r = torch.flatten(x[3], 1).float()
        # hra_i = torch.flatten(x[4], 1).float()
        # hur_r = torch.flatten(x[5], 1).float()
        # hur_i = torch.flatten(x[6], 1).float()
        # hua_r = torch.flatten(x[7], 1).float()
        # hua_i = torch.flatten(x[8], 1).float()

        x_in = torch.cat((Wr_opt, Wi_opt, h_r, h_i), 1)
        W_lin = self.linear_W(x_in)
        Wr = W_lin[:, 0*self.K_UE*self.M_AP:1*self.K_UE*self.M_AP]
        Wi = W_lin[:, 1*self.K_UE*self.M_AP:2*self.K_UE*self.M_AP]

        UL_in = torch.cat((Wr, Wi, h_r, h_i), 1)
        UL_out = self.linear_UL(UL_in)
        Ur = UL_out[:, 0*self.K_UE:1*self.K_UE]
        Ui = UL_out[:, 1*self.K_UE:2*self.K_UE]
        L = torch.abs(UL_out[:, 2*self.K_UE:3*self.K_UE])
        U = Ur + 1j*Ui

        nsr = torch.eye(self.M_AP, device=device) * 10 ** (-trainparams['snr_dB'] / 10)
        nsr = nsr.reshape((1, self.M_AP, self.M_AP))
        nsr.repeat(theta.shape[0], 1, 1)
        S = torch.zeros((theta.shape[0], self.M_AP, self.M_AP), dtype=torch.cfloat, device=device)
        W_out = torch.zeros((theta.shape[0], self.M_AP, self.K_UE), dtype=torch.cfloat, device=device)
        for j in range(self.K_UE):
            HHj = torch.einsum("bp, bq -> bpq", torch.conj(H[:,j,:]), H[:,j,:]) # m x m correlation matrix
            Uj = U[:,j]
            Lj = L[:,j]
            S += torch.square(torch.abs(Uj[:,None,None])) * Lj[:,None,None] * (nsr + HHj)
        for k in range(self.K_UE):
            Uk = U[:,k]
            Lk = L[:,k]
            W_out[:, :, k] = Uk[:,None] * Lk[:,None] * torch.linalg.solve(S, torch.conj(H[:,k,:]))

        # normfactor = torch.linalg.matrix_norm(W_out, ord='fro')
        # W_out = W_out / normfactor[:, None, None] # normalize W
        Wr_out = torch.real(W_out)
        Wi_out = torch.imag(W_out)

        return Wr_out, Wi_out

# [1] Z. Li, H. Shen, W. Xu, D. Chen, and C. Zhao, “Deep Learning-Based Adaptive Phase Shift Compression and Feedback in
# IRS-Assisted Communication Systems,” IEEE Wirel. Commun. Lett., vol. 13, no. 3, pp. 766–770, 2024,
# doi: 10.1109/LWC.2023.3342900.
class ACFNetCompressionNetwork(nn.Module):
    def __init__(self, N_RIS, N_enc):
        super(ACFNetCompressionNetwork, self).__init__()
        self.N_RIS = N_RIS
        self.convolution_layer = nn.Sequential(
            nn.Conv2d(1, 3, 2, padding=0),
            nn.ReLU(),
        )
        self.dense_layer = nn.Sequential(
            nn.Linear(297, N_enc),
        )
    def forward(self, theta):
        theta_c = torch.exp(1j * theta) # complex
        theta_r = torch.real(theta_c)
        theta_i = torch.imag(theta_c)
        theta_cnn = torch.cat((theta_r, theta_i), 1)
        theta_cnn = torch.reshape(theta_cnn, (-1, 1, self.N_RIS, 2)).float()
        theta_lin = self.convolution_layer(theta_cnn)
        theta_lin = torch.flatten(theta_lin, start_dim=1)
        theta_enc = self.dense_layer(theta_lin)
        return theta_enc

class ACFNetDecoderNetwork(nn.Module):
    def __init__(self, N_RIS, N_enc):
        super(ACFNetDecoderNetwork, self).__init__()
        self.N_RIS = N_RIS
        self.dense_layer1 = nn.Sequential(
            nn.Linear(N_enc, 2*N_RIS),
        )
        self.convolution_layer = nn.Sequential(
            nn.Conv2d(1, 3, 2, padding=0),
            nn.ReLU(),
        )
        self.dense_layer2 = nn.Sequential(
            nn.Linear(513, 2*N_RIS),
        )
    def forward(self, theta_qnt):
        theta_cnn = self.dense_layer1(theta_qnt)
        theta_cnn = torch.reshape(theta_cnn, (-1, 1, 2*trainparams['Nh_RIS'], trainparams['Nw_RIS'])).float()
        theta_lin = self.convolution_layer(theta_cnn)
        theta_lin = torch.flatten(theta_lin, start_dim=1)
        theta_ri = self.dense_layer2(theta_lin)
        theta_r = theta_ri[:, 0*self.N_RIS:1*self.N_RIS]
        theta_i = theta_ri[:, 1*self.N_RIS:2*self.N_RIS]
        theta_c = theta_r + 1j*theta_i # complex
        theta_c = theta_c / torch.abs(theta_c) # normalized
        theta_out = torch.angle(theta_c)
        return theta_out

# Inspired by: N. Shlezinger and Y. C. Eldar, “Deep task-based quantization,” Entropy, vol. 23, no. 1, pp. 1–18, Jan.
# 2021, doi: 10.3390/e23010104.
# Code: https://github.com/arielamar123/ADC-Learning-hyperopt
class AutoQEncoder(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, Nc_enc, C_code_words, dev):
        super(AutoQEncoder, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.encoder_layer = EncoderLayer(K_UE, M_AP, N_RIS, Nc_enc).to(dev)
        self.quantizer_layer = QuantizerLayer(C_code_words, dev).to(dev)
        self.decoder_layer = DecoderLayer(N_RIS, Nc_enc).to(dev)
        self.w_update_layer = WupdateLayer(K_UE, M_AP).to(dev)
    def forward(self, x):
        theta_enc = self.encoder_layer(x)
        theta_qnt = self.quantizer_layer(theta_enc)
        theta_dec = self.decoder_layer(theta_qnt)
        Wr_u, Wi_u = self.w_update_layer(theta_dec, x)
        W = Wr_u + 1j*Wi_u
        theta_out, W_out = normalizethetaW(theta_dec, W)
        return theta_out.double(), W_out.cdouble()

class AutoQEncoderNoWupdate(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, Nc_enc, C_code_words, dev):
        super(AutoQEncoderNoWupdate, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.encoder_layer = EncoderLayer(K_UE, M_AP, N_RIS, Nc_enc).to(dev)
        self.quantizer_layer = QuantizerLayer(C_code_words, dev).to(dev)
        self.decoder_layer = DecoderLayer(N_RIS, Nc_enc).to(dev)
    def forward(self, x):
        theta_enc = self.encoder_layer(x)
        theta_qnt = self.quantizer_layer(theta_enc)
        theta_dec = self.decoder_layer(theta_qnt)
        W = x[1] + 1j*x[2]
        W = torch.reshape(W, (-1, self.M_AP, self.K_UE))
        theta_out, W_out = normalizethetaW(theta_dec, W)
        return theta_out.double(), W_out.cdouble()

# [1] W. Xu, L. Gan, and C. Huang, “A Robust Deep Learning-Based Beamforming Design for RIS-Assisted Multiuser MISO
# Communications With Practical Constraints,” IEEE Trans. Cogn. Commun. Netw., vol. 8, no. 2, pp. 694–706, Jun. 2022,
# doi: 10.1109/TCCN.2021.3128605.
class DQNN(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, C_code_words, dev):
        super(DQNN, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.DNN_layer = DNN(K_UE, M_AP, N_RIS).to(dev)
        self.quantizer_layer = QuantizerLayer(C_code_words, dev).to(dev)
    def forward(self, x):
        theta, W = self.DNN_layer(x)
        theta_qnt = self.quantizer_layer(theta)
        W = torch.reshape(W, (-1, self.M_AP, self.K_UE))
        theta_out, W_out = normalizethetaW(theta_qnt, W)
        return theta_out.double(), W_out.cdouble()

# [1] Z. Li, H. Shen, W. Xu, D. Chen, and C. Zhao, “Deep Learning-Based Adaptive Phase Shift Compression and Feedback in
# IRS-Assisted Communication Systems,” IEEE Wirel. Commun. Lett., vol. 13, no. 3, pp. 766–770, 2024,
# doi: 10.1109/LWC.2023.3342900.
class ACFNet(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, N_enc, C_code_words, dev):
        super(ACFNet, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.mapping_network = DNN(K_UE, M_AP, N_RIS).to(dev)
        self.compression_network = ACFNetCompressionNetwork(N_RIS, N_enc).to(dev)
        self.quantizer_layer = QuantizerLayer(C_code_words, dev).to(dev)
        self.decoder_network = ACFNetDecoderNetwork(N_RIS, N_enc).to(dev)
    def forward(self, x):
        theta, W = self.mapping_network(x)
        W = torch.reshape(W, (-1, self.M_AP, self.K_UE))
        theta_enc = self.compression_network(theta)
        theta_qnt = self.quantizer_layer(theta_enc)
        theta_dec = self.decoder_network(theta_qnt)
        theta_out, W_out = normalizethetaW(theta_dec, W)
        return theta_out.double(), W_out.cdouble()


class LinearQuantizer(nn.Module):
    def __init__(self, K_UE, M_AP, N_RIS, Nc_enc, C_code_words, dev):
        super(LinearQuantizer, self).__init__()
        self.K_UE = K_UE
        self.M_AP = M_AP
        self.precoder_layer = nn.Linear(2*K_UE*M_AP, 2*K_UE*M_AP).to(dev)
        self.encoder_layer = nn.Linear(N_RIS, Nc_enc).to(dev)
        self.quantizer_layer = QuantizerLayer(C_code_words, dev).to(dev)
        self.decoder_layer = nn.Linear(Nc_enc, N_RIS).to(dev)
    def forward(self, x):
        theta   = x[0].float()
        theta_enc = self.encoder_layer(theta.float())
        theta_qnt = self.quantizer_layer(theta_enc)
        theta_dec = self.decoder_layer(theta_qnt)
        W_r   = x[1].float()
        W_i   = x[2].float()
        W_r   = torch.flatten(W_r, start_dim=1)
        W_i   = torch.flatten(W_i, start_dim=1)
        W_in = torch.cat((W_r, W_i), 1)
        W_out = self.precoder_layer(W_in)
        Wr = W_out[:, :(self.K_UE*self.M_AP)]
        Wi = W_out[:, (self.K_UE*self.M_AP):2*(self.K_UE*self.M_AP)]
        W = Wr + 1j*Wi
        W = torch.reshape(W, (-1, self.M_AP, self.K_UE))
        theta_out, W_out = normalizethetaW(theta_dec, W)
        return theta_out.double(), W_out.cdouble()

class LoadData(Dataset):
    def __init__(self, data):
        super(LoadData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
# https://rockt.ai/2018/04/30/einsum
# Hau = (K, M)    AP  -> UE
# Har = (N, M)    AP  -> RIS
# Hru = (K, N)    RIS -> UE
# W   = (M, K)    UE stream -> AP
def normalizethetaW(theta, W):
    theta = torch.angle(torch.exp(1j*theta))
    normfactor = torch.linalg.matrix_norm(W, ord='fro')
    W = W / normfactor[:, None, None]
    return theta, W

def batchSumRate(theta, W, Hau, Har, Hru):
    theta, W = normalizethetaW(theta, W)
    n = torch.ones(theta.shape[0], device=device) * 10**(trainparams['N_dBm']/10)
    W = W * 10**(trainparams['P_dBm']/20)
    R = torch.zeros((theta.shape[0], trainparams['K_UE']), device=device)
    for k in range(trainparams['K_UE']):
        hruHar = torch.einsum("bn, bnm -> bnm", Hru[:,k,:], Har)
        hk = Hau[:, k, :] + torch.einsum("bn, bnm -> bm", torch.exp(1j*theta), hruHar)
        S = torch.pow(torch.abs(torch.einsum("bm, bm -> b", hk, W[:, :, k])), 2)
        N = torch.zeros(theta.shape[0], device=device)
        for l in range(trainparams['K_UE']):
            if l != k:
                N += torch.pow(torch.abs(torch.einsum("bm, bm -> b", hk, W[:, :, l])), 2)
        N += n
        R[:,k] = torch.log2(torch.ones(theta.shape[0], device=device) + torch.div(S, N))
    return torch.einsum("bk -> b", R)

def Loss(theta, W, Hau, Har, Hru):
    return -torch.sum(batchSumRate(theta, W, Hau, Har, Hru))

class Trainer(object):
    def __init__(self, train_loader, trainparams, model, dev):
        self.train_loader = train_loader
        self.N_RIS = trainparams['N_RIS']
        self.Nc_enc = int(self.N_RIS * trainparams['Nc_enc_compressed_ratio'])
        self.C_code_words = trainparams['C_code_words']
        self.device = dev
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=trainparams['lr'], amsgrad=True)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=trainparams['lr'], amsgrad=True)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=trainparams['lr'], momentum=trainparams['momentum'])
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=trainparams['max_lr'], steps_per_epoch=len(train_loader),
        #                                                      epochs=trainparams['epochs'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=int(trainparams['epoch_patience']),
                                                                    eps=5e-5,
                                                                    factor=0.8)

    def train(self, val_loader, trainparams):

        val_loss_min = np.inf
        val_loss_min_earlystop = np.inf

        model_validated = deepcopy(self.model)  # if val loss does not decrease, return the copy of AQEnet before training
        train_losses = []
        val_losses = []
        num_epochs = 0
        running_loss = 0.0
        for epoch in tqdm(range(trainparams['epochs']), disable=DISABLE_TQDM):
            train_loss = 0.0
            val_loss = 0.0

            # Train the model
            self.model.train()
            with torch.enable_grad():
                try:
                    self.model.quantizer_layer.hardQ = False
                except AttributeError:
                    self.model.module.quantizer_layer.hardQ = False
                for i, data in (enumerate(self.train_loader)):
                    input, theta_opt, Wopt, Hau, Har, Hru = data
                    self.optimizer.zero_grad()                              # clear gradients of all variables to optimize
                    theta_out, W_out = self.model(input)                    # forward pass inputs into network
                    loss = Loss(theta_out, W_out, Hau, Har, Hru)            # calculate batch loss
                    loss.backward()                                         # back propagate gradients through AQE network
                    self.optimizer.step()                                   # single optimization step to update variables
                    # self.scheduler.step()                                 # One Cycle LR adjust learning rate each step
                    train_loss += loss.item()                               # update training loss
                    train_loss /= len(self.train_loader.dataset)            # normalize training loss


            # Validate the model
            self.model.eval()
            with torch.no_grad():
                # replace trainable tanh quantization layer with proper quantization layer
                try:
                    self.model.quantizer_layer.hardQ = True
                except AttributeError:
                    self.model.module.quantizer_layer.hardQ = True

                for i, data in (enumerate(val_loader)):
                    input, theta_opt, Wopt, Hau, Har, Hru = data
                    theta_out, W_out = self.model(input)                # forward pass inputs into AQE network
                    loss = Loss(theta_out, W_out, Hau, Har, Hru)        # calculate batch loss
                    val_loss += loss.item()                             # update validation loss
                    val_loss /= len(val_loader.dataset)                 # normalize validation loss

                before_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)                           # decay learning rate
                after_lr = self.optimizer.param_groups[0]['lr']
                print(before_lr, '->', after_lr, flush=True)

                if trainparams['epoch_echo']:
                    print('Epoch: {} \tTraining Loss: {:.10f} \tValidation Loss: {:.10f} \tBest Validation Loss: {:.10f}'.format(
                        epoch, train_loss, val_loss, val_loss_min), flush=True)

                # save model if validation loss has decreased
                if val_loss <= val_loss_min:
                    if trainparams['epoch_echo']:
                        print('### Validation loss same/decreased ({:.10f} --> {:.10f})... saving model. ###'.format(
                            val_loss_min, val_loss), flush=True)
                    val_loss_min = val_loss
                    model_validated = deepcopy(self.model) # if val loss does not decrease, return the copy of AQEnet before training
                # else:
                #     self.model = AQEnet_not_trained
                running_loss += val_loss
                if epoch % trainparams['epoch_val'] == trainparams['epoch_val']-1:
                    if running_loss < val_loss_min_earlystop:
                        if trainparams['epoch_echo']:
                            print('Validation early stop running loss decreased ({:.10f} --> {:.10f}).'.format(
                                val_loss_min_earlystop, running_loss), flush=True)
                        val_loss_min_earlystop = running_loss
                    else:
                        if trainparams['epoch_echo']:
                            print('No Validation early stop loss decrease ({:.10f} --> {:.10f}). Early Stopping'.format(
                            val_loss_min_earlystop, running_loss), flush=True)
                        break
                    running_loss = 0.0

                train_losses.append(train_loss)
                val_losses.append(val_loss)
            num_epochs += 1
        self.model = model_validated
        return self.model, train_losses, val_losses, num_epochs

    def evaluate(self, test_loader, trainparams):
        N_RIS = trainparams['N_RIS']
        R_opt = torch.zeros(1).to(self.device)
        R_opt_rand = torch.zeros(1).to(self.device)
        R = torch.zeros(1).to(self.device)
        R_rand = torch.zeros(1).to(self.device)
        test_size = len(test_loader.dataset.data)
        self.model.eval()
        with torch.no_grad():
            # replace trainable tanh quantization layer with proper quantization layer
            try:
                self.model.quantizer_layer.hardQ = True
            except AttributeError:
                self.model.module.quantizer_layer.hardQ = True

            for i, data in tqdm(enumerate(test_loader), disable=DISABLE_TQDM):
                input, theta_opt, Wopt, Hau, Har, Hru = data
                batchsize = theta_opt.shape[0]
                theta_model, W_model = self.model(input)  # forward pass inputs into AQE network
                theta_rand = torch.rand(size=(batchsize,N_RIS), dtype=torch.double, device=self.device) * 2*torch.pi - torch.pi

                R_opt       += torch.sum(batchSumRate(theta_opt, Wopt, Hau, Har, Hru))
                R_opt_rand  += torch.sum(batchSumRate(theta_rand, Wopt, Hau, Har, Hru))
                R           += torch.sum(batchSumRate(theta_model, W_model, Hau, Har, Hru))
                R_rand      += torch.sum(batchSumRate(theta_rand, W_model, Hau, Har, Hru))

        # Achievable Rate
        R_opt       = R_opt / test_size
        R_opt_rand  = R_opt_rand / test_size
        R           = R / test_size
        R_rand      = R_rand / test_size
        R = torch.nan_to_num(R, nan=0)
        return R_opt.item(), R_opt_rand.item(), R.item(), R_rand.item()


if __name__ == "__main__":
    print('------------')
    print('Start Script')
    print('------------')

    PdBm_dir = '40PdBm/'
    path_dir = "MATLAB/"
    trial = "00"
    dataset_dir = path_dir + "datasets/HDRISData/" + PdBm_dir
    results_dir = path_dir + "logs/MU-MISO_AchievableRateExperiments/" + trial + "/" + PdBm_dir
    # if len(sys.argv) > 1:
    #     results_dir = results_dir + sys.argv[1] + "/"
    results_file = "results.csv"
    trainparams_file = "trainparams.csv"

    print("results directory:", results_dir)
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Make print statements go to file instead of stdout:
    if path_dir == "/home/alex96/scratch/":
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        f_python_output = open(results_dir + "python_log.out", 'w')
        sys.stdout = f_python_output
        sys.stderr = f_python_output

    ####################################################################################################################
    # Training trainparams
    ####################################################################################################################
    trainparams = {'train_test_split': 0.8, # split between train/test data
                  'train_val_split': 0.8,  # after the train/test split, split train data into train/val data
                  'lr': 0.001, #10**(-1*np.random.uniform(2, 5)), # optimizer learning rate
                  # 'momentum': 0.9, # optimizer momentum for SGD
                  'batch_size': 128, #2**np.random.randint(7, 11), # batch training size
                  'epochs': 1000,  # total training duration
                  'epoch_val': 50, # validate early stop every epoch number
                  'epoch_patience': 20, # number of epochs before loss decrease
                  'epoch_echo': True, # flag to display epoch print losses
                  # 'step_size': 10, # step size for scheduler optimizer
                  # 'Nc_enc': 100, # number of quantizers, values that N is compressed/encoded into
                  'Q_bits': 1, # number of bits of a quantizer
                  # 'max_lr': 1, # maximum learning rate for Scheduler
                  }
    # for all numpy random generators, the range is: [low, high) where the low value is included but the high value is excluded.


    # print('Using OneCycleLR Scheduler, with SGD.')
    print('Using ADAM with learning rate decay')

    # Nc_array = 2**np.array(range(1,8))
    # Nc_array = [8,16,32,64,100,128]
    # Nc_array = [100]
    Nc_array = 10 * np.array(range(1,11))

    num_dirs = 1 # number of directories to use which includes data samples

    ####################################################################################################################
    # Load RIS data from .csv files
    ####################################################################################################################
    print('---------')
    print('Load Data')
    print('---------')
    trainparams['mc_runs'] = 0

    for d in range(num_dirs):
        sysmodelparams = pd.read_csv(dataset_dir + str(d) + "/" + "systemModelParameters.csv").iloc[0]
        trainparams['mc_runs'] += int(sysmodelparams['mc_runs'])
        trainparams['N_RIS'] = int(sysmodelparams['N'])
        trainparams['Nw_RIS'] = int(sysmodelparams['Nw'])
        trainparams['Nh_RIS'] = int(sysmodelparams['Nh'])
        trainparams['M_AP'] = int(sysmodelparams['M'])
        trainparams['K_UE'] = int(sysmodelparams['K'])
        trainparams['P_dBm'] = sysmodelparams['PdBm']
        trainparams['N_dBm'] = sysmodelparams['NdBm']
        trainparams['snr_dB'] = sysmodelparams['SNRdB']

    print("Training Model parameters:", flush=True)
    pprint.pprint(trainparams)

    # Hau = (K, M)    AP  -> UE
    # Har = (N, M)    AP  -> RIS
    # Hru = (K, N)    RIS -> UE
    # W   = (M, K)    UE stream -> AP
    Hau = torch.zeros(trainparams['mc_runs'], trainparams['K_UE']*trainparams['M_AP'], dtype=torch.cdouble)
    Har = torch.zeros(trainparams['mc_runs'], trainparams['N_RIS']*trainparams['M_AP'], dtype=torch.cdouble)
    Hru = torch.zeros(trainparams['mc_runs'], trainparams['K_UE']*trainparams['N_RIS'], dtype=torch.cdouble)
    RISopt = torch.zeros(trainparams['mc_runs'], trainparams['N_RIS'], dtype=torch.double)
    Wopt = torch.zeros(trainparams['mc_runs'], trainparams['M_AP']*trainparams['K_UE'], dtype=torch.cdouble)

    for d in range(num_dirs):
        print(dataset_dir + str(d) + "/", flush=True)
        sp = pd.read_csv(dataset_dir + str(d) + "/" + "systemModelParameters.csv").iloc[0]
        print('Loading... (Hau)', flush=True)
        Hau_ = load_complex(dataset_dir + str(d) + "/", "Hau_r", "Hau_i")
        Hau[d*int(sp['mc_runs']):(d+1)*int(sp['mc_runs']), :] = torch.from_numpy(Hau_)
        print('Loading... (Har)', flush=True)
        Har_ = load_complex(dataset_dir + str(d) + "/", "Har_r", "Har_i")
        Har[d*int(sp['mc_runs']):(d+1)*int(sp['mc_runs']), :] = torch.from_numpy(Har_)
        print('Loading... (Hru)', flush=True)
        Hru_ = load_complex(dataset_dir + str(d) + "/", "Hru_r", "Hru_i")
        Hru[d*int(sp['mc_runs']):(d+1)*int(sp['mc_runs']), :] = torch.from_numpy(Hru_)
        print('Loading... RISopt.csv', flush=True)
        RISopt_ = np.loadtxt(dataset_dir + str(d) + "/" + "RISopt.csv", delimiter=',')
        RISopt[d*int(sp['mc_runs']):(d+1)*int(sp['mc_runs']), :] = torch.from_numpy(RISopt_)
        print('Loading... (wopt)', flush=True)
        wopt_ = load_complex(dataset_dir + str(d) + "/", "wopt_r", "wopt_i")
        Wopt[d*int(sp['mc_runs']):(d+1)*int(sp['mc_runs']), :] = torch.from_numpy(wopt_)

    # unvectorize by reshaping the stacked vector into Matrix along proper dimensions
    Hau = torch.transpose(torch.reshape(Hau, (trainparams['mc_runs'], trainparams['M_AP'], trainparams['K_UE'])), 1, 2).to(device)
    Har = torch.transpose(torch.reshape(Har, (trainparams['mc_runs'], trainparams['M_AP'], trainparams['N_RIS'])), 1, 2).to(device)
    Hru = torch.transpose(torch.reshape(Hru, (trainparams['mc_runs'], trainparams['N_RIS'], trainparams['K_UE'])), 1, 2).to(device)
    RISopt = RISopt.to(device)
    Wopt = torch.transpose(torch.reshape(Wopt, (trainparams['mc_runs'], trainparams['K_UE'], trainparams['M_AP'])), 1, 2).to(device)

    ################################################################################################################
    # Create the Torch Dataset
    ################################################################################################################
    print('--------------')
    print('Create Dataset')
    print('--------------')
    num_train_val = int(trainparams['train_test_split'] * trainparams['mc_runs'])
    num_test = trainparams['mc_runs'] - num_train_val
    num_train = int(trainparams['train_val_split'] * num_train_val)
    num_val = num_train_val - num_train
    train_set = [] # [[inputs0, labels0], [inputs1, labels1], ... ]
    test_set = []
    val_set = []
    for i in tqdm(range(0, trainparams['mc_runs']), disable=DISABLE_TQDM):
        input = [RISopt[i],
                 torch.real(Wopt[i,:,:]), torch.imag(Wopt[i,:,:]),
                 torch.real(Har[i,:,:]),  torch.imag(Har[i,:,:]),
                 torch.real(Hru[i,:,:]),  torch.imag(Hru[i,:,:]),
                 torch.real(Hau[i,:,:]),  torch.imag(Hau[i,:,:])]
        data = [input, RISopt[i], Wopt[i,:,:], Hau[i,:,:], Har[i,:,:], Hru[i,:,:]]
        if i < num_train:
            train_set.append(data)
        elif i >= num_train_val:
            test_set.append(data)
        else:
            val_set.append(data)
    train_set = LoadData(train_set)
    test_set = LoadData(test_set)
    val_set = LoadData(val_set)

    ################################################################################################################
    # Train & Test model with specific hyperparameters
    ################################################################################################################
    print('------------')
    print('Train & Test')
    print('------------')
    R_opt_array = np.zeros(len(Nc_array))
    R_AQE_array = np.zeros(len(Nc_array))
    R_AQEnoW_array = np.zeros(len(Nc_array))
    R_ACF_array = np.zeros(len(Nc_array))
    R_DQNN_array = np.zeros(len(Nc_array))
    R_linQ_array = np.zeros(len(Nc_array))
    R_opt_rand_array = np.zeros(len(Nc_array))
    R_AQE_rand_array = np.zeros(len(Nc_array))
    R_AQEnoW_rand_array = np.zeros(len(Nc_array))
    R_ACF_rand_array = np.zeros(len(Nc_array))
    R_DQNN_rand_array = np.zeros(len(Nc_array))
    R_linQ_rand_array = np.zeros(len(Nc_array))
    trainparamslist = []
    for i in range(len(Nc_array)):
        trainparams['Nc_enc'] = Nc_array[i]
        bits = trainparams['Q_bits'] # bits per Quantizer
        trainparams['C_code_words'] = 2**bits
        trainparams['Nc_enc_compressed_ratio'] = trainparams['Nc_enc'] / trainparams['N_RIS']
        trainparams['overall_bits'] = trainparams['Nc_enc'] * bits
        train_loader = DataLoader(train_set, batch_size=trainparams['batch_size'])
        test_loader = DataLoader(test_set, batch_size=trainparams['batch_size'])
        val_loader = DataLoader(val_set, batch_size=trainparams['batch_size'])

        AQEnet = AutoQEncoder(trainparams['K_UE'], trainparams['M_AP'], trainparams['N_RIS'], trainparams['Nc_enc'], trainparams['C_code_words'], device)
        AQEnoWnet = AutoQEncoderNoWupdate(trainparams['K_UE'], trainparams['M_AP'], trainparams['N_RIS'], trainparams['Nc_enc'], trainparams['C_code_words'], device)
        ACFnet = ACFNet(trainparams['K_UE'], trainparams['M_AP'], trainparams['N_RIS'], trainparams['Nc_enc'], trainparams['C_code_words'], device)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            dqnn = DQNN(trainparams['K_UE'], trainparams['M_AP'], trainparams['N_RIS'], trainparams['C_code_words'], device)
        linQ = LinearQuantizer(trainparams['K_UE'], trainparams['M_AP'], trainparams['N_RIS'], trainparams['Nc_enc'], trainparams['C_code_words'], device)

        AQEtrainer = Trainer(train_loader, trainparams, AQEnet, device)
        total_params = sum(p.numel() for p in AQEtrainer.model.parameters())
        print('AQE Number of parameters:', total_params, flush=True)
        print(AQEtrainer.model, flush=True)

        AQEnoWtrainer = Trainer(train_loader, trainparams, AQEnoWnet, device)
        total_params = sum(p.numel() for p in AQEnoWtrainer.model.parameters())
        print('AQEnoW Number of parameters:', total_params, flush=True)
        print(AQEnoWtrainer.model, flush=True)

        ACFtrainer = Trainer(train_loader, trainparams, ACFnet, device)
        total_params = sum(p.numel() for p in ACFtrainer.model.parameters())
        print('ACF Number of parameters:', total_params, flush=True)
        print(ACFtrainer.model, flush=True)

        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            dqnntrainer = Trainer(train_loader, trainparams, dqnn, device)
            total_params = sum(p.numel() for p in dqnntrainer.model.parameters())
            print('DQNN Number of parameters:', total_params, flush=True)
            print(dqnntrainer.model, flush=True)

        linQtrainer = Trainer(train_loader, trainparams, linQ, device)
        total_params = sum(p.numel() for p in linQtrainer.model.parameters())
        print('linQ Number of parameters:', total_params, flush=True)
        print(linQtrainer.model, flush=True)

        AQEnet,     AQEnet_train_losses,    AQEnet_val_losses,  AQEnet_num_epochs   = AQEtrainer.train(val_loader, trainparams)
        AQEnoWnet,     AQEnoWnet_train_losses,    AQEnoWnet_val_losses,  AQEnoWnet_num_epochs   = AQEnoWtrainer.train(val_loader, trainparams)
        ACFnet,     ACFnet_train_losses,    ACFnet_val_losses,  ACFnet_num_epochs   = ACFtrainer.train(val_loader, trainparams)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            dqnn,       DQNNnet_train_losses,   DQNNnet_val_losses, DQNNnet_num_epochs  = dqnntrainer.train(val_loader, trainparams)
        linQ,       linQ_train_losses,      linQ_val_losses,    linQ_num_epochs     = linQtrainer.train(val_loader, trainparams)

        d_AQE = {"train": AQEnet_train_losses, "val": AQEnet_val_losses}
        AQE_loss_df = pandas.DataFrame(d_AQE)
        d_AQEnoW = {"train": AQEnoWnet_train_losses, "val": AQEnoWnet_val_losses}
        AQEnoW_loss_df = pandas.DataFrame(d_AQEnoW)
        d_ACF = {"train": ACFnet_train_losses, "val": ACFnet_val_losses}
        ACF_loss_df = pandas.DataFrame(d_ACF)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            d_DQNN = {"train": DQNNnet_train_losses, "val": DQNNnet_val_losses}
            DQNN_loss_df = pandas.DataFrame(d_DQNN)
        d_linQ = {"train": linQ_train_losses, "val": linQ_val_losses}
        linQ_loss_df = pandas.DataFrame(d_linQ)

        loss_file = "loss" + str(i) + ".csv"
        print("Saving losses to:", results_dir + "AQE_" + loss_file, flush=True)
        AQE_loss_df.to_csv(results_dir + "AQE_" + loss_file, sep='\t', encoding='utf-8', index=False, header=True)
        print("Saving losses to:", results_dir + "AQEnoW_" + loss_file, flush=True)
        AQEnoW_loss_df.to_csv(results_dir + "AQEnoW_" + loss_file, sep='\t', encoding='utf-8', index=False, header=True)
        print("Saving losses to:", results_dir + "ACF_" + loss_file, flush=True)
        ACF_loss_df.to_csv(results_dir + "ACF_" + loss_file, sep='\t', encoding='utf-8', index=False, header=True)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            print("Saving losses to:", results_dir + "DQNN_" + loss_file, flush=True)
            DQNN_loss_df.to_csv(results_dir + "DQNN_" + loss_file, sep='\t', encoding='utf-8', index=False, header=True)
        print("Saving losses to:", results_dir + "linQ_" + loss_file, flush=True)
        linQ_loss_df.to_csv(results_dir + "linQ_" + loss_file, sep='\t', encoding='utf-8', index=False, header=True)

        R_opt, R_opt_rand, R_AQE, R_AQE_rand = AQEtrainer.evaluate(test_loader, trainparams)
        R_opt, R_opt_rand, R_AQEnoW, R_AQEnoW_rand = AQEnoWtrainer.evaluate(test_loader, trainparams)
        R_opt, R_opt_rand, R_ACF, R_ACF_rand = ACFtrainer.evaluate(test_loader, trainparams)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            R_opt, R_opt_rand, R_DQNN, R_DQNN_rand = dqnntrainer.evaluate(test_loader, trainparams)
        R_opt, R_opt_rand, R_linQ, R_linQ_rand = linQtrainer.evaluate(test_loader, trainparams)

        print("-------------------------------------------------------------------------------------------------------")
        print("Training Model parameters:", flush=True)
        pprint.pprint(trainparams)
        print('Achievable Rate (bps/Hz) using RIS phase shifts with Transmit power SNR {:.2f} dB:'.format(trainparams['snr_dB']), flush=True)
        print('optimum:   ', R_opt, flush=True)
        print('AQE net:   ', R_AQE, flush=True)
        print('AQEnoW net:   ', R_AQEnoW, flush=True)
        print('ACF net:   ', R_ACF, flush=True)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            print('DQNN net:  ', R_DQNN, flush=True)
        print('linQ net:  ', R_linQ, flush=True)
        print('opt rand:  ', R_opt_rand, flush=True)
        print('AQE rand:  ', R_AQE_rand, flush=True)
        print('AQEnoW rand:  ', R_AQEnoW_rand, flush=True)
        print('ACF rand:  ', R_ACF_rand, flush=True)
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            print('DQNN rand: ', R_DQNN_rand, flush=True)
        print('linQ rand: ', R_linQ_rand, flush=True)
        print("-------------------------------------------------------------------------------------------------------\n\n")
        R_opt_array[i] = R_opt
        R_AQE_array[i] = R_AQE
        R_AQEnoW_array[i] = R_AQEnoW
        R_ACF_array[i] = R_ACF
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            R_DQNN_array[i] = R_DQNN
        R_linQ_array[i] = R_linQ
        R_opt_rand_array[i] = R_opt_rand
        R_AQE_rand_array[i] = R_AQE_rand
        R_AQEnoW_rand_array[i] = R_AQEnoW_rand
        R_ACF_rand_array[i] = R_ACF_rand
        if trainparams['Nc_enc'] == trainparams['N_RIS']:
            R_DQNN_rand_array[i] = R_DQNN_rand
        R_linQ_rand_array[i] = R_linQ_rand

        # x_vals, soft_quant, hard_quant = linQ.quantizer_layer.plot_vals()
        # fig, ax = plt.subplots()
        # ax.plot(x_vals, soft_quant, label='Soft Quantizer')
        # ax.plot(x_vals, hard_quant, label='Hard Quantizer', linestyle='--')
        # ax.set_title("Quantizer Values: bits = " + str(bits))
        # ax.legend()
        # plt.show(block=True)
        # # plt.interactive(False)

        trainparamslist.append(deepcopy(trainparams))

    trainparams_df = pandas.DataFrame(trainparamslist)
    print(trainparams_df, flush=True)
    print('Saving parameters to:', results_dir + trainparams_file)
    trainparams_df.to_csv(results_dir + trainparams_file, sep='\t', encoding='utf-8', index=False, header=True)

    d = {'Nc': Nc_array,
         'R_opt': R_opt_array,
         'R_AQE': R_AQE_array,
         'R_AQEnoW': R_AQEnoW_array,
         'R_ACF': R_ACF_array,
         'R_DQNN': R_DQNN_array,
         'R_linQ': R_linQ_array,
         'R_opt_rand': R_opt_rand_array,
         'R_AQE_rand': R_AQE_rand_array,
         'R_AQEnoW_rand': R_AQEnoW_rand_array,
         'R_ACF_rand': R_ACF_rand_array,
         'R_DQNN_rand': R_DQNN_rand_array,
         'R_linQ_rand': R_linQ_rand_array}
    results_df = pandas.DataFrame(d)
    print('Bits per Quantizer:', trainparams['Q_bits'], flush=True)
    print(results_df.to_string(), flush=True)
    print('Saving results to:', results_dir + results_file, flush=True)
    results_df.to_csv(results_dir + results_file, sep='\t', encoding='utf-8', index=False, header=True)


    ################################################################################################################
    # END PYTHON SCRIPT
    ################################################################################################################
    print('----------')
    print('End Script')
    print('----------')

    if path_dir == "/home/alex96/scratch/":
        sys.stdout = orig_stdout
        f_python_output.close()
        sys.stderr = orig_stderr