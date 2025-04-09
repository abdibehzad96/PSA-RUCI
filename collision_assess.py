# Load the predicton data from \collision_assess folder
import pandas as pd
import torch
from utilz.utils import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
Zones = [[(489,78),(549,112),(354,370),(338, 312)],
          [(580,126),(632,102),(798, 199),(788,246)],
            [(783, 246),(811,306),(669, 515),(625,498)],
            [(622, 504),(580, 535),(349,408),(363, 330)]]
def zonefinder(BB, Zones):
    z = []
    for n in range(BB.shape[0]):
        for i, zone in enumerate(Zones):
            Poly = Polygon(zone)
            if Poly.contains(Point(int(BB[n,0]), int(BB[n, 1]))):
                z.append(i+1)
                break   
    if z == []:
        return 0         
    return torch.tensor(z)

ped_frames = torch.load('data/collision_assess/ped_sara_frame.pt', weights_only=True)
ped_preds = torch.load('data/collision_assess/ped_sara_pred.pt',  weights_only=True)
ped_x = torch.load('data/collision_assess/ped_sara_x.pt',  weights_only=True)
ped_y = torch.load('data/collision_assess/ped_sara_y.pt',  weights_only=True)
device = ped_frames.device
sl = ped_frames.shape[1]//2
veh_frames = torch.load('data/collision_assess/veh_sara_frame.pt',  weights_only=True)
veh_ETA = torch.load('data/collision_assess/veh_sara_pred.pt',  weights_only=True)
veh_x = torch.load('data/collision_assess/veh_sara_x.pt',  weights_only=True)
veh_y = torch.load('data/collision_assess/veh_sara_y.pt',  weights_only=True)
column_order = [ 'x', 'y',  'kortright_light', 'gordon_light', 'path', 'zone','yield', 'object type' ,
                'angle', 'dx',  'dy', 'distance', 'speed']
veh_column_order = [ 'Frame Number', 'ETA','real_ETA', 'Path']
veh_cat = torch.cat((veh_frames, veh_ETA[:,1:-1].to(torch.int),veh_y, veh_x[:,:,4:5]), dim=-1).cpu()
Vehicles = pd.DataFrame(veh_cat.reshape(-1,veh_cat.shape[-1]).cpu().numpy(),columns=veh_column_order)
ped_column_order = ['Frame Number','x','y','zone']

New_pred_pos = ped_preds[:,sl-1:] + ped_frames[:,0:1,:2] # global positions
ped_cat = torch.cat((ped_frames[:,sl:,-1:], New_pred_pos, ped_x[:,:,-1:]) , dim=-1).cpu()
real_ped_data = torch.cat((ped_frames[:,sl:], ped_x[:,:,-1:]) , dim=-1).cpu()
# Pedestrians = pd.DataFrame(ped_cat.reshape(-1,ped_cat.shape[-1]).cpu().numpy(), columns=ped_column_order)
Conflicts_zone = {1:[0,1,3,4,9,10,11], 2:[2,6,10,11,12,13], 3:[0,1,2,3,5,7,9,13], 4:[4,5,6,7,8,12]}
# Conflicts_path = {0:[3,1], 1:[3,1], 2:[3,2], 3:[1,3], 4:[1,4], 5:[3,4], 6:[4,2], 7:[4,3], 8:[4,1], 9:[1,3], 10:[1,2], 11:[2,1], 12:[2,4], 13:[2,3]}
# Paths_xy = {0:[[743,335],[465,157]], 1:[[767,301],[485,125]], 2:[[743,302],[746,193]], 3:[[393,256],[668,457]], 4:[[374,283],[405,374]],
#             5:[[718,378],[460,412]], 6:[[536,462],[743,193]], 7:[[566,488],[668,457]], 8:[[507, 445],[465,157]], 9:[[412,231],[688,432]],
#             10:[[430,202],[728,185]], 11:[[652,140],[485,125]], 12:[[674,151],[460,412]], 13:[[713,175],[688,432]]}
Paths_xy = {1:[[743,335],[465,157]], 2:[[767,301],[485,125]], 3:[[743,302],[746,193]], 4:[[393,256],[668,457]], 6:[[374,283],[405,374]],
            7:[[718,378],[460,412]], 8:[[536,462],[743,193]], 9:[[566,488],[668,457]], 10:[[507, 445],[465,157]], 11:[[412,231],[688,432]],
            12:[[430,202],[728,185]], 13:[[652,140],[485,125]], 14:[[674,151],[460,412]], 15:[[713,175],[688,432]]} # Sara's pathing system
Conflicts_path = {1:[3,1], 2:[3,1], 3:[3,2], 4:[1,3], 6:[1,4], 7:[3,4], 8:[4,2], 9:[4,3], 10:[4,1], 11:[1,3], 12:[1,2], 13:[2,1], 14:[2,4], 15:[2,3]} # Sara's pathing system
history = 3
S_th = 2
Danger_frames = veh_cat[:,:,:2].sum(dim=-1)
real_Danger_frames = veh_cat[:,:,[0,2]].sum(dim=-1)

total_near_misses = []
total_real_Near_misses = []

for R_th in range(30, 100, 5):
    Near_misses = []
    real_Near_misses = []
    for n, danger in enumerate(Danger_frames): # For now just check the last frame
        # check if a pedestrian is present in the danger zone
        danger_frame = torch.arange(danger[-history:].min()-10, danger[-history:].max()+10)
        real_Danger_frame = torch.arange(real_Danger_frames[n,-history:].min()-10, real_Danger_frames[n, -history:].max()+10)
        peds_in_danger = ped_cat[torch.isin(ped_cat[:,-history:,0], danger_frame).any(1)]
        real_peds_in_danger = real_ped_data[torch.isin(real_ped_data[:,-history:,2], real_Danger_frame).any(1)]
        if peds_in_danger.any(): # So far we only checjerking if there is a pedestrian in the frame when the vehicle reaches the crosswalk
            Path = veh_cat[n,-1,-1].tolist()
            conflicts = torch.tensor(Conflicts_path[Path])

            for ped_in_danger in peds_in_danger: # We check if the paths conflict or not
                # Now we should check the distance
                Zone_Ped = zonefinder(ped_in_danger[-history:,1:3], Zones)
                confs = torch.nonzero(torch.isin(conflicts, Zone_Ped))
                for conf in confs:
                    # conf = check_confs(confs)
                    xref = Paths_xy[Path][conf][0]
                    yref = Paths_xy[Path][conf][1]
                    R = distance(ped_in_danger[-history:,1], ped_in_danger[-history:,2], xref, yref)
                    if (R < R_th).any():
                        Near_misses.append([n,ped_in_danger, veh_cat[n]])
                        break
        if real_peds_in_danger.any():
            Path = veh_cat[n,-1,-1].tolist()
            conflicts = torch.tensor(Conflicts_path[Path])
            for real_ped_in_danger in real_peds_in_danger: # We check if the paths conflict or not
                # Now we should check the distance
                real_Zone_Ped = zonefinder(real_ped_in_danger[-history:,:2], Zones)
                confs = torch.nonzero(torch.isin(conflicts, real_Zone_Ped))
                for conf in confs:
                    xref = Paths_xy[Path][conf][0]
                    yref = Paths_xy[Path][conf][1]
                    R = distance(real_ped_in_danger[-history:,0], real_ped_in_danger[-history:,1], xref, yref)
                    if (R < R_th).any():

                        real_Near_misses.append([n,real_ped_in_danger])
                        break
    print('Near Misses', len(Near_misses))
    print('Real Near Misses', len(real_Near_misses))
    total_near_misses.append(Near_misses)
    total_real_Near_misses.append(real_Near_misses)
print("done")
import pickle


with open("total_near_misses.pkl", "wb") as f:
    pickle.dump(total_near_misses, f)


with open("total_real_Near_misses.pkl", "wb") as f:
    pickle.dump(total_real_Near_misses, f)
print("done")
j = 0
count = 0
for n in range(29, 100, 9):
    
    k = 0
    for m , ped in total_near_misses[j]:
        if m != total_real_Near_misses[j][k][0]:
            print("Inconsistency Detected", m , total_real_Near_misses[j][k][0])
            count += 1
        k +=1
    j += 1
print(count)
print("done")
                    




