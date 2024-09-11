import sys
import pickle
from typing import Union
sys.path.insert(0,'../../')
from multihole_probe import NeuralNet_MultiLayer, KAN
import numpy.typing as npt
import numpy as np 
import torch 
import pandas as pd 
import matplotlib.pyplot as plt

def predict_data(FAP1:Union[float,npt.NDArray],FAP2:Union[float,npt.NDArray],
                 FAP3:Union[float,npt.NDArray],FAP4:Union[float,npt.NDArray],
                 FAP5:Union[float,npt.NDArray], Ps:Union[float,npt.NDArray]):
    """Predicts the data 

    Args:
        FAP1 (Union[float,npt.NDArray]): Hole 1, Center Hole
        FAP2 (Union[float,npt.NDArray]): Hole 2
        FAP3 (Union[float,npt.NDArray]): Hole 3
        FAP4 (Union[float,npt.NDArray]): Hole 4
        FAP5 (Union[float,npt.NDArray]): Hole 5
        Ps (Union[float,npt.NDArray]): Static Pressure
    """
    # Process the Data 
    data = pickle.load(open('checkpoints/16328-1_Trained_Models.pkl','rb'))
    model1_minmax = data['model1']['minmax']
    model2_minmax = data['model2']['minmax']
    Pavg = 0.25 * (FAP2 + FAP3 + FAP4 + FAP5)
    Cp1 = (FAP2-Pavg)/(FAP1-Pavg) 
    Cp2 = (FAP3-Pavg)/(FAP1-Pavg)
    Cp3 = (FAP4-Pavg)/(FAP1-Pavg)
    Cp4 = (FAP5-Pavg)/(FAP1-Pavg)
    Cpm = (FAP1-Pavg)/(FAP1)
    
    input = torch.tensor(np.stack([Cp1,Cp2,Cp3,Cp4,Cpm,Cp1*0,Cp1*0]).transpose())
    model1_minmax.fit(input)
    input = input.float()
    
    # Build the model 
    model_data = pickle.load(open('checkpoints/16328-1_Trained_Models.pkl.pkl','rb'))

    nn_model1 = NeuralNet_MultiLayer(model_data['in_features'],
                        model_data['hidden_layers'],
                        model_data['out_features'])
    nn_model2 = NeuralNet_MultiLayer(model_data['in_features'],
                        model_data['hidden_layers'],
                        model_data['out_features'])
    
    kan_layers = [model_data['in_features']]; kan_layers.extend(model_data['hidden_layers']); kan_layers.append(model_data['out_features'])
    kan_model1 = KAN(kan_layers)
    kan_model2 = KAN(kan_layers)
    
    nn_model1.load_state_dict(model_data['nn_model1'])
    nn_model2.load_state_dict(model_data['nn_model2'])
    kan_model1.load_state_dict(model_data['KAN_model1'])
    kan_model2.load_state_dict(model_data['KAN_model2'])
    
    # Predict the Data 
    nn_out1 = nn_model1(input[:,:model_data['in_features']])
    nn_out2 = nn_model2(input[:,:model_data['in_features']])
    kan_out1 = kan_model1(input[:,:model_data['in_features']])
    kan_out2 = kan_model2(input[:,:model_data['in_features']])
    
    # Scale data back 
    input[:,-2:] = nn_out1
    nn_out1 = model1_minmax.inverse_transform(input.detach().numpy())
    input[:,-2:] = nn_out2
    nn_out2 = model2_minmax.inverse_transform(input.detach().numpy())
    
    input[:,-2:] = kan_out1
    kan_out1 = model1_minmax.inverse_transform(input.detach().numpy())
    input[:,-2:] = kan_out2
    kan_out2 = model2_minmax.inverse_transform(input.detach().numpy())
    
    return nn_out1,nn_out2,kan_out1,kan_out2
    
def plot(out1:npt.NDArray,out2:npt.NDArray,network_name:str):
    """Plot the errors

    Args:
        out1 (npt.NDArray): Output from model 1
        out2 (npt.NDArray): Output from model 2
        model_name (str): Name of the neural network 
    """
    df = pd.read_csv('../../dataset/16328-1_Trained_Models.pkl')
    probe_loc = [0,1805]
    df = df.iloc[probe_loc[0]:probe_loc[1]]
    FAP2 = df['FAP2'].values	# Hole 5, Center Hole 
    Ps = df['PSFJ'].values
    theta = df['YAW'].values
    phi = df['PITCH'].values
    mach = df['MFJ'].values
    Pt = df['PTFJ'].values
    
    Pt_predict = (FAP2-Ps) / out2[:,-1] + Ps
    # Plot Relative Percent difference
    # https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
    theta_err = 2*(theta - out1[:,-2])/(theta+out1[:,-2])
    phi_err = 2*(phi - out1[:,-1])/(phi + out1[:,-1])
    
    mach_err = (mach - out2[:,-2])/mach
    Pt_err = (Pt - Pt_predict)/Pt
    
    # Plot of Error 
    plt.figure(num=1,clear=True,figsize=(10,8))
    plt.tricontourf(theta,phi,theta_err,levels=11)
    plt.xlabel('Theta')
    plt.ylabel('Phi')
    plt.title(f'{network_name} Theta Relative Percent Difference Error')
    plt.colorbar()
    plt.rcParams.update({'font.size': 18})
    plt.savefig(f'results/{network_name}_theta_error.png',dpi=300,transparent=None)

    plt.figure(num=2,clear=True,figsize=(10,8))
    plt.tricontourf(theta,phi,phi_err,levels=11)
    plt.xlabel('Theta')
    plt.ylabel('Phi')
    plt.title(f'{network_name} Phi Relative Percent Difference Error')
    plt.colorbar()
    plt.rcParams.update({'font.size': 18})
    plt.savefig(f'results/{network_name}_Phi_error.png',dpi=300,transparent=None)

    plt.figure(num=3,clear=True,figsize=(10,8))
    plt.tricontourf(theta,phi,mach_err,levels=11)
    plt.xlabel('Theta')
    plt.ylabel('Phi')
    plt.title(f'{network_name} Mach Percent Error')
    plt.colorbar()
    plt.rcParams.update({'font.size': 18})
    plt.savefig(f'results/{network_name}_Mach_error.png',dpi=300,transparent=None)
    
    plt.figure(num=4,clear=True,figsize=(10,8))
    plt.tricontourf(theta,phi,Pt_err,levels=11)
    plt.xlabel('Theta')
    plt.ylabel('Phi')
    plt.title(f'{network_name} Pt Percent Error')
    plt.colorbar()
    plt.rcParams.update({'font.size': 18})
    plt.savefig(f'results/{network_name}_Pt_error.png',dpi=300,transparent=None)

    
if __name__ == "__main__":
    probe = '16328-1_Trained_Models.pkl'
    # Fit all the data 
    df = pd.read_csv('../../dataset/16328-1_Trained_Models.pkl')
    probe_loc = [0,1805]
    df = df.iloc[probe_loc[0]:probe_loc[1]]
    FAP2 = df['FAP2'].values	# Hole 5, Center Hole 
    FAP3 = df['FAP3'].values	# Hole 1
    FAP4 = df['FAP4'].values    # Hole 2
    FAP5 = df['FAP5'].values    # Hole 3
    FAP6 = df['FAP6'].values    # Hole 4
    Ps = df['PSFJ'].values
    
    nn_out1,nn_out2,kan_out1,kan_out2 = predict_data(FAP2,FAP3,FAP4,FAP5,FAP6,Ps)
    
    plot(nn_out1,nn_out2,'Multi-Layer Perception')
    plot(kan_out1,kan_out2,'KAN')

    