import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.signal as signal
import torch
import os
from torch.utils.data import Dataset
def fast_occmap(df):
    occmap = np.zeros(len(df))
    X = df['x']
    Y = df['y'] # Zero is reserved for no zone
    for i, n in enumerate(X.index):
        x = X[n]
        y = Y[n]
        if y >= -13 and y <= 6:
            if x >= -30 and x <= -10:
                occmap[i] = 1 # inside intersection
            elif x >= -65 and x <= -30:
                occmap[i] = 2 # left side of the intersection
            elif x >= -10 and x <= 23:
                occmap[i] = 3 # right side of the intesection

        elif y >= -83 and y <= -13:
            if x >= -31 and x <= -9:
                occmap[i]= 5 # lower bound

        elif y >= 6 and y <= 76:
            if x >= -31 and x <= -9:
                occmap[i] = 4 # upper bound
    df['zone'] = occmap
    return df
def path_id(df, class_lables):
    df = calculate_avg_speed(df)
    n = 0
    zones = df['zone']
    appeared_zones = pd.unique(zones)
    if class_lables is not None: # This only happens for the SinD
    # check if appeared_zones has 0 in it then drop it only in SinD
        appeared_zones = appeared_zones[appeared_zones != 0]
    if len(appeared_zones) != 3 and class_lables is not None:
        df['path'] = 0
        print('Path is not complete')
    else:
        if class_lables is not None:
            df['path'] = class_lables.checkpair(appeared_zones)
        second_zone = (zones != appeared_zones[0]).sum() # find the point the vehicle entered the second zone
        lastzone = (zones != appeared_zones[-1]).sum() # find the point the vehicle entered the third zone
        order = np.ones_like(zones) # ordering the points where the vehicle was in each zone
        order[second_zone:] += 1
        order[lastzone:] +=1
        df['zone_order'] = order # smth like 1....1 - 2...2 - 3...3
        time2next_zone = np.zeros_like(zones)
        time2next_zone[:second_zone] = np.arange(second_zone,0,-1)
        time2next_zone[second_zone:lastzone] = np.arange(lastzone-second_zone,0,-1)
        df['t2nz'] = time2next_zone
    return df

def add_trflight(df, traffic_light):
    # Trf 1 is for zones 2, 3, 1
    # Trf 2 is for zones 4 , 5, 1
    # 0 Red, 1 Green, 3 Yellow ::: we added 1 to it ::: now is 1 red, 2 Green, 4 Yellow
    Trf = traffic_light[df['frame_id'].to_numpy()]+1
    Zone = df['zone'].to_numpy()
    Trf1 = (Zone == 2) + (Zone == 3) + (Zone == 1)
    Trf2 = (Zone == 4) + (Zone == 5) + (Zone == 1)
    df['trflightA'] = Trf[:,0] * Trf1
    df['trflightB'] = Trf[:,1] * Trf2
    return df


    
class pairs:
    def __init__(self, path):
        # check if the pair is already exist in the path
        if os.path.exists(path):
            self.pair = np.load(path, allow_pickle=True).item()
        else:
            self.pair = {}
            self.pair[0] = 0
    def save_pair(self, path):
        np.save(path, self.pair)

    def checkpair(self,x):
        for n in self.pair:
            if (self.pair[n] == x).all():
                return n
        self.pair[n+1] = x
        return n+1
    
def slicing(x, sl, sw, indx):
    sliced_df = []
    zone_order = x['zone_order']
    second_zone = (zone_order==1).sum()
    last_zone = (zone_order!=3).sum()
    # x = priority(x, class_counter)
    x_dropped = x.drop(columns =['Detected Object ID'])
    for seq in range(0,second_zone-sl, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        new_df['x_to_be_avgd'] = x_dropped['x'].iloc[second_zone]
        new_df['y_to_be_avgd'] = x_dropped['y'].iloc[second_zone]
        indx.increment(1)
        sliced_df.append(new_df)
    
    for seq in range(second_zone,last_zone-sl, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        new_df['x_to_be_avgd'] = x_dropped['x'].iloc[last_zone]
        new_df['y_to_be_avgd'] = x_dropped['y'].iloc[last_zone]
        indx.increment(1)
        sliced_df.append(new_df)
    if sliced_df:
        return pd.concat(sliced_df, ignore_index=True)
    else:
        return None
def slicing_Kort_Gord(x, sl, sw, indx):
    sliced_df = []
    zone_order = x['zone_order']
    second_zone = (zone_order==1).sum()
    last_zone = (zone_order!=3).sum()
    x_dropped = x.drop(columns =['Detected Object ID'])
    for seq in range(0,second_zone-sl-sw, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        indx.increment(1)
        sliced_df.append(new_df)
    for seq in range(second_zone,last_zone-sl-sw, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        indx.increment(1)
        sliced_df.append(new_df)
    if sliced_df:
        return pd.concat(sliced_df, ignore_index=True)
    else:
        return None
    
def slicing_sind(x, sl, sw, indx):
    sliced_df = []
    zone_order = x['zone_order']
    second_zone = (zone_order==1).sum()
    last_zone = (zone_order!=3).sum()
    # x = priority(x, class_counter)
    x_dropped = x.drop(columns =['track_id','frame_id','timestamp_ms'])
    for seq in range(0,second_zone-sl, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        new_df['x_to_be_avgd'] = x_dropped['x'].iloc[second_zone]
        new_df['y_to_be_avgd'] = x_dropped['y'].iloc[second_zone]
        indx.increment(1)
        sliced_df.append(new_df)
    
    for seq in range(second_zone-sl,last_zone-sl, sw):
        new_df = x_dropped[seq: seq+sl]
        new_df['indx'] = indx.indx
        new_df['x_to_be_avgd'] = x_dropped['x'].iloc[last_zone]
        new_df['y_to_be_avgd'] = x_dropped['y'].iloc[last_zone]
        indx.increment(1)
        sliced_df.append(new_df)
    if sliced_df:
        return pd.concat(sliced_df, ignore_index=True)
    else:
        return None
    
class index:
    def __init__(self):
        self.indx = 0
    def increment(self,v):
        self.indx +=v

def priority(x, class_counter):
    priority = np.ones(len(x))
    leading = np.ones(len(x))
    paths = x['path']
    i = 0
    for n, path_ego in paths.items():
        for m, path_surr in paths.items():
            if path_surr == class_counter[path_ego]:
                R = distance(x['x'][n], x['y'][n], x['x'][m], x['y'][m])
                if R < 40:
                    priority[i] = 0
                    
            elif path_surr == path_ego and n != m:
                # R = distance(x['x'][n], x['y'][n], x['x'][m], x['y'][m])
                Rx = x['x'][n] - x['x'][m]
                Ry = x['y'][n] - x['y'][m]
                if Rx < 10 or Ry < 10:
                    leading[i] = check_leading(Rx, Ry, x['zone'][n])
        i += 1
    x['priority'] = priority
    x['leading'] = leading
    return x
    
def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def check_leading(Rx, Ry, zone):
    if zone == 1:
        return 2
    elif zone == 2:
        return int(Rx > 0 or np.abs(Ry) > 3)
    elif zone == 3:
        return int(Rx < 0 or np.abs(Ry) > 3)
    elif zone == 4:
        return int(Ry < 0 or np.abs(Rx) > 3)
    elif zone == 5:
        return int(Ry > 0 or np.abs(Rx) > 3)
    else:
        return 1

def create_dataset(seq_len, sw, trafic_light, column_order, data_path = None, class_pair_path = None, class_counter_path = None):
    if data_path is None:
        data_path = r'sind\Changchun\changchun_pudong_507_009\Veh_smoothed_tracks.csv'
    data = pd.read_csv(data_path)
    class_lables = pairs(class_pair_path)
    class_counter = np.load(class_counter_path, allow_pickle=True).item()
    indx = index()
    zone_added = data.groupby('track_id', as_index=False).apply(fast_occmap).reset_index(drop=True) #.apply(filter_before_and_after_first_c_zone).reset_index(drop=True)
    traffic_added = zone_added.groupby('frame_id').apply(lambda x: add_trflight(x, trafic_light)).reset_index(drop=True)
    path_added = traffic_added.groupby('track_id', as_index=False).apply(lambda x: path_id(x,class_lables)).reset_index(drop=True)
    agent_added = path_added[path_added['path']!=0] # They are the ones that only appeared in two zones
    # removed_non_movement = movementadded.groupby('track_id', as_index=False).filter(lambda x: (x['movement'] != 0).any()).reset_index(drop=True)
    # Encode categorical features
    le_agent = LabelEncoder()
    agent_added.loc[:,'agent_type'] = le_agent.fit_transform(agent_added['agent_type'])
    Priority_added = agent_added.groupby('frame_id').apply(lambda x: priority(x, class_counter)).reset_index(drop=True)
    sliced = Priority_added.groupby('track_id', as_index=False).apply(lambda x: slicing_sind(x, seq_len, sw, indx)).reset_index(drop=True)
    # indices = list(range(total_data_len))
    # random.shuffle(indices)
    AvgAdded = sliced.groupby('path').apply(lambda x: avg_zone(x))
    AvgAdded['xd2nz'] = np.abs((AvgAdded['x'] - AvgAdded['avgx']))
    AvgAdded['yd2nz'] = np.abs((AvgAdded['y'] - AvgAdded['avgy']))
    y_path = AvgAdded.groupby('indx', as_index=False)['path'].first().to_numpy()[:,1]# We only need one value of the movement, as it is constant in the whole sequence
    y_time = AvgAdded.groupby('indx', as_index=False).apply(lambda x: x['t2nz']).to_numpy()
    x_path = AvgAdded.groupby('indx').apply(lambda x: x[column_order]).reset_index(drop=True).to_numpy()
    if not os.path.exists(class_pair_path):
        class_lables.save_pair(class_pair_path)
    return x_path, y_path, y_time

def avg_zone(x):
    new_df = []
    for z in pd.unique(x['zone']):
        A = x[x['zone']==z]
        A['avgx'] = A['x_to_be_avgd'].mean()
        A['avgy'] = A['y_to_be_avgd'].mean()
        new_df.append(A)
    return pd.concat(new_df, ignore_index=True)
def read_light(path, maxframe):

    df_light = pd.read_csv(path)
    light_tensor = torch.zeros(maxframe+101, 2)
    light_dict = {}
    memory = (0,0)
    frame = 0
    flag = 0

    for row in df_light.itertuples():
        if row[1] < frame:
            memory = row[3:]
            continue
        while frame < row[1]:
            light_dict[frame] = memory
            light_tensor[frame] = torch.tensor(memory)
            frame += 1
            if frame > maxframe + 100:
                flag = 1
                break
        memory = row[3:]
        if flag == 1:
            break

    return light_dict, light_tensor


def ped_dataset(seq_len, sw, trafic_light, column_order, data_path = None):
    if data_path is None:
        data_path = r'sind\Changchun\changchun_pudong_507_009\Veh_smoothed_tracks.csv'
    data = pd.read_csv(data_path)
    indx = index()
    zone_added = data.groupby('track_id', as_index=False).apply(fast_occmap).reset_index(drop=True)
    traffic_added = zone_added.groupby('frame_id').apply(lambda x: ped_trflight(x, trafic_light)).reset_index(drop=True)
    sliced = traffic_added.groupby('track_id', as_index=False).apply(lambda x: ped_slicing(x, seq_len, sw, indx)).reset_index(drop=True)
    processed = sliced.groupby('indx').apply(lambda x: x[column_order]).reset_index(drop=True)
    return processed.to_numpy()

def init_point(df):
    df['dx'] = df['x'] - df['x'].iloc[0]
    df['dy'] = df['y'] - df['y'].iloc[0]
    # df['init_x'] = df['x'].iloc[0]
    # df['init_y'] = df['y'].iloc[0]
    return df

def ped_trflight(df, traffic_light):
    # Trf 1 is for zones 2, 3, 1
    # Trf 2 is for zones 4 , 5, 1
    # 0 Red, 1 Green, 3 Yellow ::: we added 1 to it ::: now is 1 red, 2 Green, 4 Yellow
    Trf = traffic_light[df['frame_id'].to_numpy()]
    df['trflightA'] = Trf[:,0]
    df['trflightB'] = Trf[:,1]
    return df

def ped_slicing(x, sl, sw, indx):
    sliced_df = []
    len_x = len(x)
    # x = priority(x, class_counter)
    x_dropped = x.drop(columns =['frame_id','track_id','timestamp_ms', 'agent_type'])
    for seq in range(0,len_x-2*sl, sw):
        new_df = x_dropped[seq: seq+2*sl]
        new_df['indx'] = indx.indx
        new_df = init_point(new_df)
        indx.increment(1)
        sliced_df.append(new_df)

    if sliced_df:
        return pd.concat(sliced_df, ignore_index=True)
    else:
        return None

class peddataset(Dataset):
    def __init__(self, x, y,frame, ds, device):
        self.x = x[:,::ds].to(device).to(torch.float)
        self.y = y[:,::ds].to(device).to(torch.float)
        self.frame = frame[:,::ds].to(device).to(torch.float)
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
    
def create_dataset_Kort_Gord(seq_len, sw, column_order, data_path, class_pair_path, class_counter_path):
    Z1 = ['C1','Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    Z2 = ['C2', 'Z6', 'Z7', 'Z8']
    Z3 = ['C3','Z9', 'Z10', 'Z11', 'Z12', 'Z13']
    Z4 = ['C4','Z14', 'Z15', 'Z16', 'Z17', 'Z18']
    Cs = [1,2,3,4]
    data = pd.read_csv(data_path)
    data.loc[data['zone'].isin(Z1), 'zone'] = 1
    data.loc[data['zone'].isin(Z2), 'zone'] = 2
    data.loc[data['zone'].isin(Z3), 'zone'] = 3
    data.loc[data['zone'].isin(Z4), 'zone'] = 4
    data.loc[~data['zone'].isin(Cs), 'zone'] = 5
    if class_pair_path is not None:
        class_lables = pairs(class_pair_path)
        # class_counter = np.load(class_counter_path, allow_pickle=True).item()
    else:
        class_lables = None
    indx = index()
    le_agent = LabelEncoder()
    # le_zone = LabelEncoder()
    
    # data.loc[:,'zone'] = le_zone.fit_transform(data['zone'])
    # zone_enc = pd.unique(data.loc[:,'zone'])
    # zone_transform = le_zone.inverse_transform(zone_enc.tolist())
    data.loc[:,'object type'] = le_agent.fit_transform(data['object type'])

    data = data.groupby('Detected Object ID', as_index=False).apply(lambda x: path_id(x,class_lables)).reset_index(drop=True)
    sliced = data.groupby('Detected Object ID', as_index=False).apply(lambda x: slicing_Kort_Gord(x, seq_len, sw, indx)).reset_index(drop=True)

    y_path = sliced.groupby('indx', as_index=False)['path'].first().to_numpy()[:,1] # We only need one value of the movement, as it is constant in the whole sequence
    y_time = sliced.groupby('indx', as_index=False).apply(lambda x: x['t2nz']).to_numpy()
    x_path = sliced.groupby('indx').apply(lambda x: x[column_order]).reset_index(drop=True).to_numpy()
    return x_path, y_path, y_time

def calculate_avg_speed(group):
    x= group['x']
    y= group['y']
    group['dx'] = np.diff(x, append = x.iloc[-1])
    group['dy'] = np.diff(y, append = y.iloc[-1])
    return group

def ped_dataset_Kort_Gord(seq_len, sw, trafic_light, column_order, window_size, data_path = None):
    Z1 = ['C1','Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    Z2 = ['C2', 'Z6', 'Z7', 'Z8']
    Z3 = ['C3','Z9', 'Z10', 'Z11', 'Z12', 'Z13']
    Z4 = ['C4','Z14', 'Z15', 'Z16', 'Z17', 'Z18']
    Cs = [1,2,3,4]
    data = pd.read_csv(data_path)
    data.loc[data['Zone'].isin(Z1), 'Zone'] = 1
    data.loc[data['Zone'].isin(Z2), 'Zone'] = 2
    data.loc[data['Zone'].isin(Z3), 'Zone'] = 3
    data.loc[data['Zone'].isin(Z4), 'Zone'] = 4
    data.loc[~data['Zone'].isin(Cs), 'Zone'] = 5
    indx = index()
    # le_zone = LabelEncoder()
    # data.loc[:,'zone'] = le_zone.fit_transform(data['Zone'])
    # zone_enc = pd.unique(data.loc[:,'zone'])
    # zone_transform = le_zone.inverse_transform(zone_enc.tolist())
    traffic_added = data.merge(trafic_light, on='Frame Number', how='left').fillna(0)
    sliced = traffic_added.groupby('Detected Object ID', as_index=False).apply(lambda x: ped_slicing_Kort_Gord(x, seq_len, sw, window_size, indx)).reset_index(drop=True)
    processed = sliced.groupby('indx').apply(lambda x: x[column_order]).reset_index(drop=True)
    return processed.to_numpy()

def ped_slicing_Kort_Gord(df, sl, sw, window_size, indx):
    if not hasattr(ped_slicing_Kort_Gord, "value"):
        ped_slicing_Kort_Gord.filter = np.ones(window_size)/window_size
    sliced_df = []
    len_x = len(df)
    # x = priority(x, class_counter)
    if (df[['sat_y', 'sat_y']] ==0).all().all() or len_x < 2*sl:
        return None
    x_dropped = df.drop(columns =['Detected Object ID','type']) # we dont drop 'Frame Number' as we need it for the collision time calculation
    x = np.convolve(x_dropped['sat_x'], ped_slicing_Kort_Gord.filter , mode='valid')
    y = np.convolve(x_dropped['sat_y'], ped_slicing_Kort_Gord.filter , mode='valid')
    # We cut out the the last window_size -1 elements and then replace the smoothed values with the original values
    x_dropped = x_dropped.iloc[window_size//2:-(window_size//2)]
    x_dropped['sat_x'] = x 
    x_dropped['sat_y'] = y
    len_x = len(x)
    for seq in range(0,len_x-2*sl- window_size, sw):
        new_df = x_dropped[seq: seq+2*sl + window_size]
        new_df['indx'] = indx.indx
        new_df = init_point(new_df, window_size)
        indx.increment(1)
        sliced_df.append(new_df)

    if sliced_df:
        return pd.concat(sliced_df, ignore_index=True)
    else:
        return None
    
def init_point(df, window_size):
    if not hasattr(init_point, "value"):
        init_point.filter = np.ones(window_size)/window_size
    
    dx = df['sat_x']- df['sat_x'].iloc[0]
    dy = df['sat_y'] - df['sat_y'].iloc[0]
    
    dx= np.convolve(dx, init_point.filter , mode='valid')
    dy= np.convolve(dy, init_point.filter , mode='valid')
    df = df.iloc[window_size//2:-(window_size//2)]
    df['dx'] = dx
    df['dy'] = dy
    # df['init_x'] = df['x'].iloc[0]
    # df['init_y'] = df['y'].iloc[0]
    return df[:-1]

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the input data.
    
    Parameters:
    - data: Input signal (list or numpy array)
    - cutoff: Cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Filter order (higher = sharper cutoff)
    
    Returns:
    - Filtered signal (numpy array)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)  # Design filter
    filtered_data = signal.filtfilt(b, a, data)  # Apply filter
    return filtered_data