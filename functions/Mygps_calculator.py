import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.io import loadmat
from scipy import interpolate

#------------------------------------    This Part loads the data, extract the features and save them in an excel file     -----------------------------------

# ----- Acess nested structure -----
def access_struct (data,structs):

    for struct in structs:    
        if isinstance(data,np.ndarray) and data.dtype.names is not None:
            data = data[0,0][struct]
        else:
            data = data[struct]
        
    return data

# ----- Sort the order of the files -----
def natural_sort_key(s):

    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# ----- Access features in the mat files -----
def gps_features (file_path:str, output_path:str, file_num:int, num_all_files :int, separate_legs:bool):
    file = loadmat(file_path)
    side_structs = ['Right', 'Left']
    joint_names = ['Pelvis', 'Hip', 'Knee', 'Ankle', 'FootProgress']
    if separate_legs == False:
        joint_data = np.empty((100,0))

        for side_struct in side_structs:
            structs = ['c', 'results', side_struct, 'angAtFullCycle']
            all_data = access_struct(file,structs)
            side = side_struct[0]

            if side_struct == 'Left':
                joint_with_side = 'LPelvis'
                joint_kin = all_data[0,0][joint_with_side][0][0]
                joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

            for joint in (joint_names):
                if joint == 'Knee':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
                    
                elif joint == 'Ankle':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

                elif joint == 'FootProgress':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,2]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
                
                elif joint == 'Hip': # This considers hip
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0]
                    joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

        joint_data = pd.DataFrame(joint_data)
        joint_data.columns = ['R_Hip flex/ext', 'R_Hip abd/add', 'R_Hip int/ext rotation', 'R_Knee flx/ext',
                              'R_Ankle dorsi/plantar flx', 'R_foot progression', 'L_Pelvis', 'L_Pelvis',
                              'L_Pelvis', 'L_Hip flex/ext','L_Hip abd/add', 'L_Hip int/ext rotation',
                              'L_Knee flx/ext', 'L_Ankle dorsi/plantar flx', 'L_foot progression']

        if file_num < (((num_all_files)+1)//2):
            joint_data.to_csv(output_path+'\Subject%d_PreLokomat.csv' %(file_num+1), index = False)

        else:
            joint_data.to_csv(output_path+'\Subject%d_PostLokomat.csv' %(file_num + 1 - (((num_all_files)+1)//2)), index = False)

        print("The data of both sides is extracted together!")
        return joint_data
        
    
    else:
        joint_data_both_sides = pd.DataFrame()
        for side_struct in side_structs:
            joint_data = np.empty((100,0))
            structs = ['c', 'results', side_struct, 'angAtFullCycle']
            all_data = access_struct(file,structs)
            side = side_struct[0]

            for joint in (joint_names):
                if joint == 'Pelvis':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0]
                    joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

                elif joint == 'Knee':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
                    
                elif joint == 'Ankle':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

                elif joint == 'FootProgress':
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,2]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
                
                else: # This considers hip
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0]
                    joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
            
            
            joint_data = pd.DataFrame(joint_data)
            dofs = ['Pelvis tilt', 'Pelvis rot','Pelvis obli', 'Hip flex/ext',
                   'Hip abd/add', 'Hip int/ext rotation','Knee flx/ext',
                   'Ankle dorsi/plantar flx', 'foot progression']
            joint_data.columns = [side + '_' + dof for dof in dofs]
            
            if file_num < (((num_all_files)+1)//2) or file_num == 25:
                joint_data.to_csv(output_path+'\Subject%d_%s_PreLokomat.csv' %(file_num+1,side), index = False)
                print('\Subject%d_PreLokomat.csv for %s' %(file_num+1, side_struct))
            else:
                joint_data.to_csv(output_path+'\Subject%d_%s_PostLokomat.csv' %(file_num + 1 - (((num_all_files)+1)//2),side), index = False)
                print('\Subject%d_PreLokomat.csv for %s' %(file_num+1, side_struct))
            
            joint_data_both_sides = pd.concat([joint_data_both_sides, joint_data], axis=1)

        print("Subject%d" %(file_num+1))
        print("The data of each side is extracted separately!")
        return joint_data_both_sides
          
    
def calculate_gps (data, reference, separate_legs:bool):
    
    if separate_legs == False:
        all_GVS = []
        for i in range(15):
            diffrences = data.values[:,i] - reference[i]
            GVS = np.sqrt(np.mean(diffrences**2))
            all_GVS.append(GVS)

        GPS = np.mean(all_GVS)
        return GPS
        
    else:
        right_all_GVS = []; left_all_GVS = []
        right_leg = data.loc[:, data.columns.str.startswith('R')]
        left_leg = data.loc[:, data.columns.str.startswith('L')]
        reference = np.concatenate((reference[6:9], (reference[9:] + reference[:6])/2))

        for i in range(len(reference)):
            right_diffrences = right_leg.values[:,i] - reference[i]; left_diffrences = left_leg.values[:,i] - reference[i]
            right_GVS = np.sqrt(np.mean(right_diffrences**2)); left_GVS = np.sqrt(np.mean(left_diffrences**2))
            right_all_GVS.append(right_GVS); left_all_GVS.append(left_GVS)

        r_GPS = np.mean(right_all_GVS); l_GPS = np.mean(left_all_GVS)
         
        return r_GPS, l_GPS
#----------------------------------------------------------------------------

input_directory = r"/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/CPData_PrePostLokomat (matfiles)/"
pre_files = [f for f in os.listdir(input_directory) if f.endswith("eLokomat.mat")]
post_files = [f for f in os.listdir(input_directory) if f.endswith("stLokomat.mat")]
mat_files_pre_sorted = sorted(pre_files, key=natural_sort_key)
mat_files_post_sorted = sorted(post_files, key=natural_sort_key)
mat_files_sorted = mat_files_pre_sorted + mat_files_post_sorted

output_path = r"/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyScripts/Functions/GPScache"

reference = pd.read_csv(r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/TD Data (matfiles)/average.csv')
reference.drop(reference.columns[0], axis=1, inplace=True)
reference = reference.values
all_GPS = []
# ----- Read data: First read all Pre datas and the all Post datas -----
print("----------------------------------")
print("LOADING DATA")
print("----------------------------------")
for indx, file in enumerate(mat_files_sorted, start = 0):
    file_path = os.path.join(input_directory, file)
    joint_data = gps_features (file_path, output_path=output_path, file_num=indx,
                               num_all_files = len(list(enumerate(mat_files_sorted, start = 0))),
                               separate_legs=True)
    gps = calculate_gps(joint_data, reference, separate_legs=True)
    all_GPS.append(gps)


all_GPS = np.array(all_GPS)
all_GPS = all_GPS.flatten()
reshaped_all_GPS = all_GPS.reshape((-1,2), order='F')
GPS_output = pd.DataFrame(reshaped_all_GPS)
GPS_output.columns = ['Pre', 'Post']


# ----- CAUTION::: ACTIVATE THE FOLLOWING LINES WHEN 
#                  SEPARATE_LEGS = True,
#                  OTHERWISE, IGNORE IT.
row_names = ['Right', 'Left']
row_names = np.tile(row_names, len(reshaped_all_GPS)//2)
GPS_output.index = row_names
GPS_output.to_csv(r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyScripts/Functions/GPS_output26.csv')

print("----------------------------------")
print("ANALYSIS DONE!")
print("----------------------------------")
