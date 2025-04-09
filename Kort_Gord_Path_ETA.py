import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utilz.utils import *
import pickle
from torch.utils.data import DataLoader
from models.model import *
import time
Kort_Gord_or_sinD = 'Kort_Gord'
scratch = True
scratch = False
SaveModel = True  # check if we should save the model or not
Loadmodel = False # Load the pre-trained model
IgnoreDataLoader = True
RF = False
ds, sl, sw = 2, 60, 2
hidden_size, num_layer, output_size = 256, 1, 1
epoch, batch = 400, 128
path = r'data/Vehicles.csv'
# class_pair_path = r'data/class_pair.npy'
# class_counter_path = r'data/class_counter.npy'
column_order = [ 'x', 'y',  'kortright_light', 'gordon_light', 'path', 'zone','yield', 'object type',
                'angle', 'dx',  'dy', 'distance', 'speed', 'Frame Number']
frame_col = [13]
swap = True
swap_column = [0,1,2,3,4,5,6,7,8,9,10, 11, 12]
RF_column = [0,1,2,3,5,7,8, 9,10,11,12] # It should be according to the swap_column
if scratch:
    x_path0, y_path, y_time0 = create_dataset_Kort_Gord(sl, sw, column_order, path, None, None)
    input_size = x_path0.shape[1]
    # path is on index 18
    x_path = x_path0.reshape(-1, sl, input_size)
    y_time = y_time0.reshape(-1, sl)
    np.save(r"data/x_path.npy", x_path)
    np.save(r"data/y_path.npy", y_path)
    np.save(r"data/y_time.npy", y_time)
else:
    x_path = np.load(r'data/x_path.npy', allow_pickle=True)
    y_path = np.load(r'data/y_path.npy', allow_pickle=True)
    y_time = np.load(r'data/y_time.npy', allow_pickle=True)
    input_size = x_path.shape[-1]

if swap:
    frame = x_path[:, :, frame_col]
    x_path = x_path[:, :sl, swap_column]
    input_size = x_path.shape[-1]
    print("Swapped columns")
# y_pt = np.concatenate([np.expand_dims(y_path, 1), np.expand_dims(y_time, 1)], axis=1)
X_path_train, X_path_test, y_path_train, y_path_test, y_time_train, y_time_test, frame_train, frame_test = train_test_split(x_path, y_path,y_time,frame, test_size=0.2 , random_state=  42, shuffle=True)
print(" Train Size: ", X_path_train.shape, " Test Size: ", X_path_test.shape)
if RF:
    X_path_train_RF = X_path_train[:,::ds, RF_column].reshape(-1, sl*len(RF_column)//ds)
    Y_path_train_RF = y_path_train
    if Loadmodel:
        with open('randomforest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
    else:
        rf_model = RandomForestClassifier(n_estimators=20)
        rf_model.fit(X_path_train_RF, Y_path_train_RF)

    X_path_test_RF = X_path_test[:,::ds, RF_column].reshape(-1, sl*len(RF_column)//ds)
    Y_path_test_RF = y_path_test

    start_time = time.perf_counter()  # High-precision start time
    y_path_pred = rf_model.predict(X_path_test_RF)
    elapsed_time = time.perf_counter() - start_time  # High-precision elapsed time
    print(f"Elapsed time: {1000*elapsed_time/len(y_path_pred):.4f} ms")
    rf_accuracy = accuracy_score(Y_path_test_RF, y_path_pred)
    print(f"Random Forest Path Number Prediction Accuracy: {rf_accuracy}")

    if rf_accuracy > 0.9 and SaveModel:
        with open('randomforest.pkl','wb') as f:
            pickle.dump(rf_model,f)


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
train_dataset = dataset(torch.from_numpy(X_path_train.astype(float)).to(torch.float), torch.from_numpy(y_time_train.astype(float)).to(torch.float),torch.from_numpy(frame_train.astype(int)).to(torch.float), ds, device)
test_dataset = dataset(torch.from_numpy(X_path_test.astype(float)).to(torch.float), torch.from_numpy(y_time_test.astype(float)).to(torch.float),torch.from_numpy(frame_test.astype(int)).to(torch.float), ds, device)

train_loader = DataLoader(train_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)

config = {}

config['tvcv_cols'] = [0,1,2,3,4,5,6,7] #time varying categorial variables columns
config['tvlv_cols'] = torch.arange(len(config['tvcv_cols']), input_size) # time varying linear variables columns
config['time_varying_categoical_variables'] = len(config['tvcv_cols'])
config['time_varying_linear_variables'] = len(config['tvlv_cols'])
config['tv_emb_size'] =[1024, 1024, 7, 7, 18, 9, 5, 9]
config['sos'] = torch.cat((torch.tensor([1020, 1020, 4, 4, 16, 7, 2, 6]), torch.zeros(input_size-len(config['tv_emb_size'])))).to(device)
config['eos'] = torch.cat((torch.tensor([1022, 1022, 5, 5, 17, 8, 3, 7]), -torch.ones(input_size-len(config['tv_emb_size'])))).to(device)
config['embedding_dim'] = hidden_size
config['input_size'] = input_size
config['output_size'] = 1
config['hidden_size'] = hidden_size
config['num_layer'] = num_layer
config['dropout'] = 0.15
config['device'] = device
config['batch_size'] = batch
config['attn_heads'] = 4
config['seq_length'] = sl//ds +2

model = ETA(config)
model.to(device)
print("Sent model to", device)
criterion = nn.SmoothL1Loss()  # For classification tasks

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay =1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.3)

BestModel = train(model, train_loader, test_loader, criterion, optimizer, scheduler, epoch, Kort_Gord_or_sinD)

print("Training Done")
testloss = evaluate(BestModel, test_loader, config['sos'], config['eos'], Kort_Gord_or_sinD, True)

print("Evaluation Done")