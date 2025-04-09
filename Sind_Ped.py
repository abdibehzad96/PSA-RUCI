import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ast import literal_eval
from sind_utils import *
import pickle
from torch.utils.data import DataLoader
from PedETA import *
scratch = True
scratch = False
SaveModel = True  # check if we should save the model or not
Loadmodel = False # Load the pre-trained model
IgnoreDataLoader = True
RF = False
ds, sl, sw = 2, 60, 2
hidden_size, num_layer = 512, 1
epoch, batch = 400, 256
path = r'sind/Changchun/changchun_pudong_507_009/Ped_smoothed_tracks.csv'

column_order = [ 'dx', 'dy',  'trflightA', 'trflightB',
                'vx',  'vy', 'ax',  'ay', 
                'zone', 'x', 'y']
column2use = [ 0, 1 ,2, 3, 4, 5, 6, 7, 8]
column2predict = [0, 1]
real_pos_column = [9, 10]
input_size = len(column2use)
output_size = len(column2predict)

if scratch:
    _, trafic_light = read_light(r'sind/Changchun/changchun_pudong_507_009/Traffic_Lights.csv', 15396*3)
    trafic_light = trafic_light[::3]
    processed = ped_dataset(sl, sw, trafic_light, column_order, path)
    feat_size = processed.shape[1]
    # path is on index 18
    path = processed.reshape(-1, 2*sl, feat_size)
    inpt_ped = path[:, :sl, column2use]
    target_ped = path[:, sl:, column2predict]
    real_pos = path[:, :, real_pos_column]
    np.save(r"sind/Changchun/changchun_pudong_507_009/ped/Ped_input.npy", inpt_ped)
    np.save(r"sind/Changchun/changchun_pudong_507_009/ped/Ped_target.npy", target_ped)
    np.save(r"sind/Changchun/changchun_pudong_507_009/ped/Ped_real_pos.npy", real_pos)
else:
    inpt_ped = np.load(r'sind/Changchun/changchun_pudong_507_009/ped/Ped_input.npy', allow_pickle=True)
    target_ped = np.load(r'sind/Changchun/changchun_pudong_507_009/ped/Ped_target.npy', allow_pickle=True)
    real_pos = np.load(r'sind/Changchun/changchun_pudong_507_009/ped/Ped_real_pos.npy', allow_pickle=True)
    input_size = inpt_ped.shape[-1]
    output_size = target_ped.shape[-1]

# y_pt = np.concatenate([np.expand_dims(y_path, 1), np.expand_dims(y_time, 1)], axis=1)
X_path_train, X_path_test, y_path_train, y_path_test, real_pos_train, real_pos_test = train_test_split(inpt_ped, target_ped,real_pos, test_size=0.2 , random_state= 30, shuffle=True)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
train_dataset = peddataset(torch.from_numpy(X_path_train), torch.from_numpy(y_path_train),torch.from_numpy(real_pos_train), ds, device)
test_dataset = peddataset(torch.from_numpy(X_path_test), torch.from_numpy(y_path_test), torch.from_numpy(real_pos_test), ds, device)

train_loader = DataLoader(train_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)


config = {}
config['embedding_dim'] = hidden_size
config['input_size'] = input_size
config['output_size'] = output_size
config['hidden_size'] = hidden_size
config['num_layer'] = num_layer

config['dropout'] = 0.15
config['device'] = device
config['batch_size'] = batch
config['seq_length'] = sl//ds
config['horizon'] = sl//ds
# model = TFT(config)
model = PedETA(config)
# model = Prophet()
model.to(device)
print("Sent model to", device)
criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay =1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

train(model, train_loader, test_loader, criterion, optimizer, scheduler, epoch)

print("Training Done")
# evaluate(model, test_loader)

print("Evaluation Done")