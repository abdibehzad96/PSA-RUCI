import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import math
import matplotlib.pyplot as plt

class ETA(nn.Module):
    def __init__(self, config):
        super(ETA, self).__init__()
        # Embedding layers
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        self.tvcc = config['tvcv_cols']
        self.len_tvcc = len(self.tvcc)
        self.tvlc = config['tvlv_cols']
        self.sos = config['sos']
        self.eos = config['eos']
        self.time_varying_embedding_layers = nn.ModuleList()

        for i in range(len(config['tv_emb_size'])):
            emb = nn.Embedding(config['tv_emb_size'][i], config['embedding_dim'])
            self.time_varying_embedding_layers.append(emb)

        self.tvll0 = nn.LayerNorm(self.input_size - self.len_tvcc)
        self.tvll1 = nn.Linear(self.input_size - self.len_tvcc, config['embedding_dim'])
        self.tvll2= nn.Linear(config['embedding_dim'], config['embedding_dim'])
        self.LNembd = nn.LayerNorm(config['embedding_dim'])
        self.LNFC = nn.LayerNorm(self.hidden_size)

        self.BN = nn.BatchNorm2d(len(self.tvcc)+1)
        self.compressor = compressor(config['embedding_dim'], num_layer= self.num_layer, kernel_size= 3, in_channel = len(self.tvcc)+1, out_channel= 1)
        self.FC_expand = nn.Linear(config['embedding_dim'], self.hidden_size)
        self.PosEmbd = PositionalEncoding(self.hidden_size, max_len=240)
        
        self.FFMHA = FF(self.hidden_size)
        self.LNMHA = nn.LayerNorm(self.hidden_size)

        self.drp = nn.Dropout(0.05)
        self.drp2 = nn.Dropout(0.1)

        self.FCend0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.FCend2 = nn.Linear(self.hidden_size, config['output_size'])

    def forward(self, x):
        # 'x', 'y',  'trflightA', 'trflightB', 'path', 'zone', 'agent_type',
        tv_embed_var = []
        
        for i, tv_embed in enumerate(self.time_varying_embedding_layers):
            emb = tv_embed(x[:, :, self.tvcc[i]].long())
            tv_embed_var.append(emb)

        y = self.tvll0(x[:, :, self.tvlc])
        y = self.tvll1(y)
        y = self.tvll2(y) + y
        tv_embed_var.append(y)
        chanelled_embedding = torch.stack(tv_embed_var, dim=1)
        x = self.PosEmbd(chanelled_embedding)
        x = self.drp(x)
        x = self.BN(x)
        xatt = self.compressor(x).squeeze(1)
        x = self.FFMHA(xatt)
        x = self.LNMHA(self.drp2(x)) # works fine without x + xatt so don't add it
        x = self.FCend0(x)
        x = self.FCend2(F.relu(x)) # works fine with leaky relu already
        return x
    
class FF(nn.Module):
    def __init__(self, hidden_size):
        super(FF, self).__init__()
        self.FC1 = nn.Linear(hidden_size, hidden_size)
        self.Lr = nn.GELU()
        self.FC2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        x = self.FC1(x)
        x = self.Lr(x)
        x = self.FC2(x)
        return x

class compressor_unit(nn.Module):
    def __init__(self, hidden_size, in_channel, out_channel, kernel_size):
        super(compressor_unit, self).__init__()
        self.convq = nn.Conv2d(in_channel, out_channel,kernel_size, 1, padding=kernel_size//2)
        self.convk = nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=kernel_size//2)
        self.convv = nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=kernel_size//2)

        self.dropout = nn.Dropout(0.1)
        self.LN = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        q = self.convq(x)
        k = self.convk(x).transpose(-2,-1)
        v = self.convv(x)
        d_k = q.shape[-1]
        scores = torch.matmul(q, k) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = self.LN(self.dropout(output) + q) # works fine with q + output so let it be
        return output
class compressor(nn.Module):
    def __init__(self, hidden_size, num_layer, kernel_size, in_channel, out_channel):
        super(compressor, self).__init__()
        # num_layer stack of 2D convolutions with maxpooling
        assert in_channel > 2*num_layer, "Input channel must be greater than 2*num_layer"
        self.conv = nn.ModuleList([compressor_unit(hidden_size, in_channel-2*i, in_channel-2*(i+1), kernel_size) for i in range(num_layer)])
        self.lastconv = compressor_unit(hidden_size, in_channel - 2*num_layer, out_channel, kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        x = self.lastconv(x)
        return x

def train(model, traindata, testdata, criterion, optimizer,scheduler, epochs, sara_or_sinD):
    BestModel = None
    
    BestLoss = 1e6
    model.train()
    sos = model.sos
    eos = model.eos
    for epoch in range(epochs):
        save_results = False
        total_loss = 0
        true_loss = 0
        for x, y, _ in traindata:
            if sara_or_sinD == 'sind':
                x[:,:, 0] = (x[:,:,0] + 66)*2
                x[:,:, 1] = (x[:,:,1] + 82)*2
            y = y.clamp(0, 511)
            x = torch.cat((sos.repeat(x.shape[0],1,1), x, eos.repeat(x.shape[0],1,1)), dim=1)
            y = torch.cat((torch.tensor([511], device=x.device).repeat(x.shape[0],1,1), y, torch.tensor([0], device=x.device).repeat(x.shape[0],1,1)), dim=1)
            optimizer.zero_grad()
            outputs = model(x)
            # loss = criterion(outputs.reshape(-1,512), y.reshape(-1).long()) # For classification tasks
            loss = criterion(outputs, y) # For regression tasks
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError("Loss contains NaN or Inf values!")
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            # true_loss += torch.mean((outputs.softmax(-1).argmax(-1)[:,-5:-1]-y[:,-5:-1].squeeze(-1)).abs()) # For classification tasks
            true_loss += torch.mean((outputs[:,-2]-y[:,-2]).abs())   # For regression tasks
        scheduler.step()
        if epoch % 10 ==5:
            testloss = evaluate(model, testdata, sos, eos, sara_or_sinD)
            if testloss < BestLoss:
                BestLoss = testloss
                BestModel = model
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(traindata)/30:.3f} s, True Loss: {true_loss/len(traindata)/30: .4f} s")
        if save_results:
            testloss = evaluate(BestModel, testdata, sos, eos, sara_or_sinD, save_results)
        model.train()
    return BestModel


def evaluate(model, dataloader, sos, eos, sara_or_sinD, save_results=False):
    model.eval()
    testloss = 0
    elapsed_time = 0
    mse_loss = 0
    total_len = 0
    record_x = []
    record_y = []
    record_frame = []
    record_pred = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for x, y, frame in dataloader:
            y = y.clamp(0, 511)
            if save_results:
                record_x.append(x)
                record_y.append(y)
                record_frame.append(frame)
            if sara_or_sinD == 'sind':
                x[:,:, 0] = (x[:,:,0] + 66)*2
                x[:,:, 1] = (x[:,:,1] + 82)*2
            
            x = torch.cat((sos.repeat(x.shape[0],1,1), x, eos.repeat(x.shape[0],1,1)), dim=1)
            # y = torch.cat((torch.tensor(2, device=x.device).repeat(x.shape[0],1), y, torch.tensor(1, device=x.device).repeat(x.shape[0],1)), dim=1)
            outputs = model(x)
            mse_loss += F.mse_loss(outputs[:,1:-1],y)
            # testloss += torch.mean((outputs.softmax(-1).argmax(-1)[:,-5:-1]-y[:,-4:].squeeze(-1)).abs())
            testloss += torch.mean((outputs[:,-2]-y[:,-1]).abs())
            total_len += len(x)
            record_pred.append(outputs)
    torch.cuda.synchronize()  # Ensure all CUDA ops are done
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"Test Loss: {testloss/len(dataloader)/30:.3f} s, Avg MSE Loss: {torch.sqrt(mse_loss)/len(dataloader):.3f} Time: {1000*elapsed_time/total_len:.3f} us")
    if save_results:
        record_x = torch.cat(record_x, dim=0)
        record_y = torch.cat(record_y, dim=0)
        record_frame = torch.cat(record_frame, dim=0)
        record_pred = torch.cat(record_pred, dim=0)
        torch.save(record_x, f"/home/abdikhab/sara/collision_assess/veh_{sara_or_sinD}_x.pt")
        torch.save(record_y, f"/home/abdikhab/sara/collision_assess/veh_{sara_or_sinD}_y.pt")
        torch.save(record_frame, f"/home/abdikhab/sara/collision_assess/veh_{sara_or_sinD}_frame.pt")
        torch.save(record_pred, f"/home/abdikhab/sara/collision_assess/veh_{sara_or_sinD}_pred.pt")
        print("Results saved!")
    
    return testloss/len(dataloader)/30


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(2)]
        return self.dropout(x)
    
    import numpy as np
class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self):
        self.max = 1e-6
        self.min = 100
    def __call__(self, tensor):
        mx = tensor.reshape(-1, tensor.size(-1)).max(dim=0)[0]
        mn = tensor.reshape(-1, tensor.size(-1)).min(dim=0)[0]
        self.max = (mx > self.max)*mx + (mx < self.max)* self.max
        self.min = (mn < self.min)*mn + (mn > self.min)* self.min
        scale = 1.0 / (self.max - self.min+ 1e-6) 
        tensor.sub_(self.min).mul_(scale)
        return tensor
class dataset(Dataset):
    def __init__(self, x, y, frame, ds, device):
        self.x = x[:,::ds].to(device)
        self.y = y[:,::ds].unsqueeze(-1).to(device)
        self.frame = frame[:,::ds].to(device)
        self.device = device
    
    def save(self, path):
        torch.save(self.x, os.path.join(path, 'x.pt'))
        torch.save(self.y, os.path.join(path, 'y.pt'))
    
    def load(self, path):
        self.x = torch.load(os.path.join(path, 'x.pt')).to(self.device)
        self.y = torch.load(os.path.join(path, 'y.pt')).to(self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.frame[idx]
    

# Sample trajectory data
def plott(m, x):
    time = torch.arange(0,30)  # Normalized time from 0 to 1
    plt.figure()
    plt.scatter(x[m,1:-1,0].to('cpu'), x[m,1:-1,1].to('cpu'), c=time, cmap="viridis", marker="o", edgecolors="black")
    plt.plot(x[m,1:-1,0].to('cpu'), x[m,1:-1,1].to('cpu'), color="gray", linestyle="--")  # Connect points
    plt.xlim(0, 300)
    plt.ylim(0,300)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("2D Trajectory with Time Coloring")
    plt.colorbar(label="Time Progression")  # Colorbar to indicate time
    plt.show()
    plt.savefig("plot.jpg", dpi=300)