import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utilz.utils import *
from torch.utils.data import DataLoader
from models.PedETA import *
import torch.optim as optim


Kort_Gord_or_sinD = 'Kort_Gord'
scratch = True
# scratch = False
SaveModel = True  # check if we should save the model or not
Loadmodel = False # Load the pre-trained model
IgnoreDataLoader = True
RF = False
window_size = 9
ds, sl, sw = 6, 60, 10
hidden_size, num_layer = 256, 1
epoch, batch = 400, 128
path = r'data/pedestrians.csv'
pathv = r'data/Vehicles.csv'
trfL = pd.read_csv(pathv)[['Frame Number', 'kortright_light', 'gordon_light']].drop_duplicates(subset='Frame Number')
# Frame Number,Detected Object ID,sat_x,sat_y,Zone,type
column_order = [ 'dx', 'dy',  'kortright_light', 'gordon_light',
                'Zone', 'sat_x', 'sat_y', 'Frame Number']

column2use = [ 0, 1 ,2, 3, 4]
column2predict = [0, 1]
frame_col = [5,6,7]
input_size = len(column2use)
output_size = len(column2predict)

if scratch:
    # _, trafic_light = read_light(r'data/Traffic_Lights.csv', 15396*3)
    # trafic_light = trafic_light[::3]
    processed = ped_dataset_Kort_Gord(sl, sw, trfL, column_order,window_size, path)
    feat_size = processed.shape[1]
    # path is on index 18
    path = processed.reshape(-1, 2*sl, feat_size)
    inpt_ped = path[:, :sl]
    target_ped = path[:, sl:]
    frame = path[:, :, frame_col]
    np.save(r"data/ped/Ped_input.npy", inpt_ped)
    np.save(r"data/ped/Ped_target.npy", target_ped)
    np.save(r"data/ped/Ped_frames.npy", frame)
    inpt_ped = path[:, :sl, column2use]
    target_ped = path[:, sl:, column2predict]
    frame = path[:, :, frame_col]
else:
    inpt_ped = np.load(r'data/ped/Ped_input.npy', allow_pickle=True)
    target_ped = np.load(r'data/ped/Ped_target.npy', allow_pickle=True)
    frame = np.load(r'data/ped/Ped_frames.npy', allow_pickle=True)
    inpt_ped = inpt_ped[:, :, column2use]
    target_ped = target_ped[:,:, column2predict]
    input_size = inpt_ped.shape[-1]
    output_size = target_ped.shape[-1]

# y_pt = np.concatenate([np.expand_dims(y_path, 1), np.expand_dims(y_time, 1)], axis=1)
X_path_train, X_path_test, y_path_train, y_path_test, real_frame_train, real_frame_test = train_test_split(inpt_ped, target_ped,frame, test_size=0.2, shuffle=True)
print("Train_Size, Test Size", len(X_path_train), len(X_path_test))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_dataset = peddataset(torch.from_numpy(X_path_train), torch.from_numpy(y_path_train),torch.from_numpy(real_frame_train), ds, device)
test_dataset = peddataset(torch.from_numpy(X_path_test), torch.from_numpy(y_path_test), torch.from_numpy(real_frame_test), ds, device)

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

model = PedETA(config)
model.to(device)
print("Sent model to", device)
criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay =1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

BestModel = train(model, train_loader, test_loader, criterion, optimizer, scheduler, epoch)

print("Training Done")
testloss= evaluate(BestModel, test_loader, save_results = True)

print("Evaluation Done")