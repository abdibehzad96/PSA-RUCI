
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utilz.utils import *
import pickle
from torch.utils.data import DataLoader
from models.model import *
Kort_Gord_or_sinD = 'sind'
scratch = True
scratch = False
SaveModel = True  # check if we should save the model or not
Loadmodel = False # Load the pre-trained model
IgnoreDataLoader = True
RF = False
ds, sl, sw = 9, 180, 2
hidden_size, num_layer, output_size = 256, 1, 1
epoch, batch = 400, 128
path = r'sind/Changchun/changchun_pudong_507_009/Veh_smoothed_tracks.csv'
class_pair_path = r'sind/Changchun/changchun_pudong_507_009/class_pair.npy'
class_counter_path = r'sind/Changchun/changchun_pudong_507_009/class_counter.npy'
column_order = [ 'x', 'y',  'trflightA', 'trflightB', 'path', 'zone', 'agent_type','priority','leading',
                'vx',  'vy',  'yaw_rad',  'heading_rad',  'length',   'width',  'ax',   'ay',   'v_lon',   'v_lat',   'a_lon',   'a_lat', 
                'x_to_be_avgd',   'y_to_be_avgd',   'avgx',   'avgy', 'xd2nz',  'yd2nz']

swap = True
swap_column = [0,1,25,26,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20]
RF_column = [0,1,4,5,7, 8,11,12,13,14,15,16,17,18,19,20] # It should be according to the swap_column
if scratch:
    _, trafic_light = read_light(r'sind/Changchun/changchun_pudong_507_009/Traffic_Lights.csv', 15396*3)
    trafic_light = trafic_light[::3]
    x_path0, y_path, y_time0 = create_dataset(sl, sw, trafic_light, column_order, path, class_pair_path, class_counter_path)
    input_size = x_path0.shape[1]
    # path is on index 18
    x_path = x_path0.reshape(-1, sl, input_size)
    y_time = y_time0.reshape(-1, sl)
    np.save(r"sind/Changchun/changchun_pudong_507_009/x_path.npy", x_path)
    np.save(r"sind/Changchun/changchun_pudong_507_009/y_path.npy", y_path)
    np.save(r"sind/Changchun/changchun_pudong_507_009/y_time.npy", y_time)
else:
    x_path = np.load(r'sind/Changchun/changchun_pudong_507_009/x_path.npy', allow_pickle=True)
    y_path = np.load(r'sind/Changchun/changchun_pudong_507_009/y_path.npy', allow_pickle=True)
    y_time = np.load(r'sind/Changchun/changchun_pudong_507_009/y_time.npy', allow_pickle=True)
    input_size = x_path.shape[-1]

if swap:
    x_path = x_path[:, :sl, swap_column]
    input_size = x_path.shape[-1]
    print("Swapped columns")
# y_pt = np.concatenate([np.expand_dims(y_path, 1), np.expand_dims(y_time, 1)], axis=1)
X_path_train, X_path_test, y_path_train, y_path_test, y_time_train, y_time_test = train_test_split(x_path, y_path,y_time, test_size=0.2 , random_state= 30, shuffle=True)
print("SinD Train Size: ", X_path_train.shape, " Test Size: ", X_path_test.shape)
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
    y_path_pred = rf_model.predict(X_path_test_RF)
    rf_accuracy = accuracy_score(Y_path_test_RF, y_path_pred)
    print(f"Random Forest Path Number Prediction Accuracy: {rf_accuracy}")

    if rf_accuracy > 0.9 and SaveModel:
        with open('randomforest.pkl','wb') as f:
            pickle.dump(rf_model,f)

y_time_train= torch.from_numpy(y_time_train.astype(float)).to(torch.float)
y_time_test= torch.from_numpy(y_time_test.astype(float)).to(torch.float)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
train_dataset = dataset(torch.from_numpy(X_path_train.astype(float)).to(torch.float), y_time_train , torch.zeros_like(y_time_train), ds, device)
test_dataset = dataset(torch.from_numpy(X_path_test.astype(float)).to(torch.float),y_time_test ,torch.zeros_like(y_time_test), ds, device)

train_loader = DataLoader(train_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)


config = {}

config['tvcv_cols'] = [0,1,2,3,4,5,6,7,8,9,10] #time varying categorial variables columns
config['tvlv_cols'] = torch.arange(len(config['tvcv_cols']), input_size) # time varying linear variables columns
config['time_varying_categoical_variables'] = len(config['tvcv_cols'])
config['time_varying_linear_variables'] = len(config['tvlv_cols'])
config['tv_emb_size'] =[512, 512,128, 128, 8, 8, 16, 16, 11, 6, 6]
config['sos'] = torch.cat((torch.tensor([510, 510, 124, 124, 6, 6, 13, 13, 9, 4, 4]), torch.zeros(input_size-len(config['tv_emb_size'])))).to(device)
config['eos'] = torch.cat((torch.tensor([511, 511, 125, 125, 5, 5, 14, 14, 8, 3, 3]), -torch.ones(input_size-len(config['tv_emb_size'])))).to(device)
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
# model = TFT(config)
model = ETA(config)
# model = Prophet()
model.to(device)
print("Sent model to", device)
criterion = nn.SmoothL1Loss()  # For classification tasks
# criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay =1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

train(model, train_loader, test_loader, criterion, optimizer, scheduler, epoch, Kort_Gord_or_sinD)

print("Training Done")
# evaluate(model, test_loader)

print("Evaluation Done")