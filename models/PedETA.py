from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt

class PedETA(nn.Module):
    def __init__(self, config):
        super(PedETA, self).__init__()
        # Embedding layers
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        self.output_size = config['output_size']
        self.device = config['device']
        self.horizon= config['horizon']

        self.LN_input = nn.LayerNorm(self.input_size)
        self.conv1 = nn.Conv1d(self.input_size, self.input_size, 5, padding=2)
        self.tvll1 = nn.Linear(self.input_size, self.hidden_size)
        self.tvll2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config['dropout'])
        self.LNembd = nn.LayerNorm(self.hidden_size)
        self.enc_lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)
        self.dec_lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)
        # self.enc_lstm = stackResLSTM( self.hidden_size, self.num_layer, config['dropout'])
        # self.dec_lstm = stackResLSTM( self.hidden_size, self.num_layer, config['dropout'])
        self.transform0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.transform1 = nn.Linear(self.hidden_size, self.output_size)




    def forward(self, x):
        # 'x', 'y',  'trflightA', 'trflightB', 'path', 'zone', 'agent_type',
        # x = self.LN_input(x)
        x = self.conv1(x.permute(0,2,1)).permute(0,2,1) + x
        # x = self.avgpool(x)
        y = self.tvll1(x)
        y = F.leaky_relu(self.tvll2(y)) + y
        y = self.dropout(self.LNembd(y))
        y, hidden = self.enc_lstm(y)

        out = []
        nex_tocken = y[:,-1:]
        # out.append(nex_tocken)
        for i in range(1, self.horizon):
            nex_tocken, hidden = self.dec_lstm(nex_tocken, hidden)
            out.append(nex_tocken)
        out = torch.cat(out, dim=1)
        out = torch.cat((y, out), dim=1)
        out = F.leaky_relu(self.transform0(out)) + out
        out = self.transform1(out)
        return out

class stackResLSTM(nn.Module):
    def __init__(self,hidden_size, num_layers, dropout):
        super(stackResLSTM, self).__init__()
        self.LSTM = nn.ModuleList([ResLSTM(hidden_size) for _ in range(num_layers)])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, x):
        return (torch.rand(self.num_layers, x.size(0), self.hidden_size).to(x.device), torch.rand(self.num_layers, x.size(0), self.hidden_size).to(x.device))
    

    def forward(self, x, hidden = None):
        if hidden is None:
            hidden = self.init_hidden(x)
        next_out = x
        next_hs = []
        next_cs = []
        for i, lstm in enumerate(self.LSTM):
            next_out, (n_hs, n_cs) = lstm(self.dropout(next_out), (hidden[0][i:i+1], hidden[1][i:i+1]))
            next_hs.append(n_hs)
            next_cs.append(n_cs)
        next_hs = torch.cat(next_hs, dim=0)
        next_cs = torch.cat(next_cs, dim=0)
        return next_out, (next_hs, next_cs)
#lstm(next_out, x, (hidden[0][i:i+1], hidden[1][i:i+1]))

class ResLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(ResLSTM, self).__init__()
        self.LN = nn.LayerNorm(hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
    def forward(self, x , hidden):
        x = self.LN(x)
        y, hidden = self.LSTM(x, hidden)
        return y + x , hidden



def train(model, traindata, testdata, criterion, optimizer,scheduler, epochs):
    model.train()
    BestModel = model
    BestLoss = 1e6
    sl = model.horizon
    for epoch in range(epochs):
        total_loss = 0
        true_loss = 0
        deviation = 0
        save_results = False
        for x, y, _ in traindata:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs,torch.cat((x[:,1:,:2], y), dim=1)) # For regression tasks
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError("Loss contains NaN or Inf values!")
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            true_loss += torch.mean((((outputs[:,-2:].mean(-1)-y[:,-2:].mean(-1))**2).sum(-1).sqrt()))   # For regression tasks
            deviation += torch.mean((((outputs[:,-sl:]-y)**2).sum(-1).sqrt())) 
        scheduler.step()
        if epoch % 10 ==5:
            testloss = evaluate(model, testdata)
            if testloss < BestLoss:
                BestLoss = testloss
                BestModel = model 
        if save_results:
            testloss= evaluate(BestModel, testdata, save_results = True)
        model.train()
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(traindata):.3f}, True Loss: {9.02*true_loss/len(traindata): .4f} cm, deviation: {9.02*deviation/len(traindata):.3f} cm")
    return BestModel

def evaluate(model, dataloader, save_results = False):
    
    sara_or_sinD = 'sara'
    model.eval()
    sl = model.horizon
    testloss = 0
    mse_loss = 0
    deviation = 0
    total_len = 0
    record_x = []
    record_y = []
    record_frame = []
    record_pred = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
# Your computation here
    with torch.no_grad():
        for x, y, frame in dataloader:
            outputs = model(x)
            mse_loss += F.mse_loss(outputs[:,-sl:],y)
            testloss += torch.mean((((outputs[:,-2:].mean(-1)-y[:,-2:].mean(-1))**2).sum(-1).sqrt()))
            # xy = outputs.cumsum(-2) + pos[:,:1]
            deviation += torch.mean((((outputs[:,-sl:]-y)**2).sum(-1).sqrt()))
            total_len += len(x)
            record_x.append(x)
            record_y.append(y)
            record_frame.append(frame)
            record_pred.append(outputs)
    torch.cuda.synchronize()  # Ensure all CUDA ops are done
    end.record()
    record_x = torch.cat(record_x, dim=0)
    record_y = torch.cat(record_y, dim=0)
    record_frame = torch.cat(record_frame, dim=0)
    record_pred = torch.cat(record_pred, dim=0)
    torch.cuda.synchronize()  # Wait for the events to complete
    elapsed_time = start.elapsed_time(end)
    if save_results:
        torch.save(record_x, f"/home/abdikhab/sara/collision_assess/ped_{sara_or_sinD}_x.pt")
        torch.save(record_y, f"/home/abdikhab/sara/collision_assess/ped_{sara_or_sinD}_y.pt")
        torch.save(record_frame, f"/home/abdikhab/sara/collision_assess/ped_{sara_or_sinD}_frame.pt")
        torch.save(record_pred, f"/home/abdikhab/sara/collision_assess/ped_{sara_or_sinD}_pred.pt")
        print("Results saved")
    print(f"Test's Avg MSE Loss: {torch.sqrt(mse_loss)/len(dataloader):.3f}, True Loss: {9.02*testloss/len(dataloader):.3f} cm, deviation: {9.02*deviation/len(dataloader):.3f} cm, Time: {1000*elapsed_time/total_len:.3f} us")
    return testloss/len(dataloader)

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