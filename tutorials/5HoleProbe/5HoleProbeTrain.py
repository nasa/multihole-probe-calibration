import sys, os
sys.path.insert(0,'../../')
import numpy as np
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from multihole_probe import NeuralNet_MultiLayer, KAN
from multihole_probe.train_model import train_model
from sklearn.preprocessing import MinMaxScaler
import torch

def CreateCoefficients(df:pd.DataFrame):
    """Create the coefficients for model 1 and model 2

    Args:
        df (pd.DataFrame): dataframe containing probe data
    """
    FAP2 = df['FAP2'].values	# Hole 5, Center Hole 
    FAP3 = df['FAP3'].values	# Hole 1
    FAP4 = df['FAP4'].values    # Hole 2
    FAP5 = df['FAP5'].values    # Hole 3
    FAP6 = df['FAP6'].values    # Hole 4
    Pt = df['PTFJ'].values
    Ps = df['PSFJ'].values
    # Predicted 
    theta = df['YAW'].values 	# Model 1
    phi = df['PITCH'].values 	# Model 1
    mach = df['MFJ'].values		# Model 2
    Cpt = (FAP2-Ps) / (Pt-Ps) 	# Model 2
    # Used to train model to predicted values of Theta, Phi, Mach, Cpt
    Pavg = 0.25 * (FAP3 + FAP4 + FAP5 + FAP6)
    Cp1 = (FAP3-Pavg)/(FAP2-Pavg) 
    Cp2 = (FAP4-Pavg)/(FAP2-Pavg)
    Cp3 = (FAP5-Pavg)/(FAP2-Pavg)
    Cp4 = (FAP6-Pavg)/(FAP2-Pavg)
    Cpm = (FAP2-Pavg)/(FAP2)
 
    data = np.stack([Cp1,Cp2,Cp3,Cp4,Cpm,theta,phi]).transpose()
    model1_data = pd.DataFrame(data=data, index=np.arange(0,len(Cp1)), columns=['Cp1','Cp2','Cp3','Cp4','Cpm','theta','phi'])
    
    data = np.stack([Cp1,Cp2,Cp3,Cp4,Cpm,mach,Cpt]).transpose()
    model2_data = pd.DataFrame(data=data, index=np.arange(0,len(Cp1)), columns=['Cp1','Cp2','Cp3','Cp4','Cpm','mach','Cpt'])

    return model1_data, model2_data

def Step1_ProcessData():
    """Process the data into test and train for both models
    """
    # Specifies what data points to use based on calibration data. 
    # Probe caller specifies the probe. Total 10 probes
    probe_caller = 1
    if probe_caller   == 1:
        probe_loc = [0,1805] # Reading numbers corresponding to different Probes 
    elif probe_caller == 2:
        probe_loc = [1806,3609] 
    elif probe_caller == 3:
        probe_loc = [3610,5414]
    elif probe_caller == 4:
        probe_loc = [5415,7219]
    elif probe_caller == 5:
        probe_loc = [7220,9026]
    elif probe_caller == 6:
        probe_loc = [9027,10831]
    elif probe_caller == 7:
        probe_loc = [10832,12641]
    elif probe_caller == 8:
        probe_loc = [12642,14447]
    elif probe_caller == 9:
        probe_loc = [14448,16252] 
    elif probe_caller == 10:
        probe_loc = [16253,18061]
        
    #%% Read in the data 
    df = pd.read_csv('dataset/5Hole-Probe1.csv')
    # Select from datapoints 
    probe_1 = df.iloc[probe_loc[0]:probe_loc[1]]
    # validation_set = probe_1.query('MFJ<0.55 and MFJ>0.45')
    # test_train_set = probe_1.query('MFJ>0.55 or MFJ<0.45')

    model1_data,model2_data = CreateCoefficients(df=probe_1)
    
    minmax_model1 = MinMaxScaler(feature_range=(-1, 1))
    minmax_model2 = MinMaxScaler(feature_range=(-1, 1))
    minmax_model1.fit(model1_data)
    minmax_model2.fit(model2_data)
    model1_data = minmax_model1.transform(model1_data)
    model2_data = minmax_model2.transform(model2_data)
    
    train1,test1 = train_test_split(model1_data,test_size=0.2)
    train2,test2 = train_test_split(model2_data,test_size=0.2)

    
    os.makedirs('checkpoints',exist_ok=True)
    os.makedirs('results',exist_ok=True)

    pickle.dump({
                    'model1':
                        {
                            'data':model1_data, 
                            'train':train1,
                            'test':test1,
                            'minmax':minmax_model1
                        },
                    'model2':
                        {
                            'data':model2_data,
                            'train':train2,
                            'test':test2,
                            'minmax':minmax_model2
                        }
                },
                open('checkpoints/01_5HoleProbe_data.pkl','wb'))

def CreateDataLoaders(in_features:int):
    """Read in the data and create the torch data loaders 

    Args:
        in_features (int): number of input features 

    Returns:
        (Tuple) containing:
        
            *model1_train_loader* (DataLoader): DataLoader to predict Theta and Phi
            *model1_test_loader* (DataLoader): DataLoader to validate Theta and Phi
            *model2_train_loader* (DataLoader): DataLoader to predict Mach and Cpt
            *model2_test_loader* (DataLoader): DataLoader to validate Mach and Cpt
            
    """
    data = pickle.load(open('checkpoints/01_5HoleProbe_data.pkl', 'rb'))
    
    # Multi-layer perception 
    
    model1_x_train = data['model1']['train'][:,:in_features] 
    model1_y_train = data['model1']['train'][:,in_features:]
    model1_x_test = data['model1']['test'][:,:in_features] 
    model1_y_test = data['model1']['test'][:,in_features:]
    
    model2_x_train = data['model2']['train'][:,:in_features] 
    model2_y_train = data['model2']['train'][:,in_features:]
    model2_x_test = data['model2']['test'][:,:in_features] 
    model2_y_test = data['model2']['test'][:,in_features:]
    
    model1_train_dataset = torch.utils.data.TensorDataset(torch.tensor(model1_x_train).float(),torch.tensor(model1_y_train).float())
    model1_test_dataset = torch.utils.data.TensorDataset(torch.tensor(model1_x_test).float(),torch.tensor(model1_y_test).float())
    
    model2_train_dataset = torch.utils.data.TensorDataset(torch.tensor(model2_x_train).float(),torch.tensor(model2_y_train).float())
    model2_test_dataset = torch.utils.data.TensorDataset(torch.tensor(model2_x_test).float(),torch.tensor(model2_y_test).float())
    
    model1_train_loader = torch.utils.data.DataLoader(dataset=model1_train_dataset,batch_size=64)
    model1_test_loader = torch.utils.data.DataLoader(dataset=model1_test_dataset,batch_size=64)
    
    model2_train_loader = torch.utils.data.DataLoader(dataset=model2_train_dataset,batch_size=64)
    model2_test_loader = torch.utils.data.DataLoader(dataset=model2_test_dataset,batch_size=64)
    return model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader
    
def TrainMultiLayer(in_features:int,out_features:int,hidden_layers:int):
    """Train a standard DNN Network 

    Args:
        in_features (int): Number of inputs
        out_features (int): Number of outputs 
        hidden_layers (int): Size of hidden layers. Example, [8,8,8,8] 4 layers with 8 neurons per layer 


    Returns:
        Dict: Dictionary containing the model states and training and validation loss history
        
    """
    nn_model1 = NeuralNet_MultiLayer(in_features,hidden_layers,out_features)
    nn_model2 = NeuralNet_MultiLayer(in_features,hidden_layers,out_features)
    
    model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader = CreateDataLoaders(in_features)
    
    nn_model1,training_loss_history1,validation_loss_history1 = train_model(nn_model1,model1_train_loader,model1_test_loader,num_epochs=2000,verbose=True)
    nn_model2,training_loss_history2,validation_loss_history2 = train_model(nn_model2,model2_train_loader,model2_test_loader,num_epochs=2000,verbose=True)
    
    return {'nn_model1': nn_model1.state_dict(),
            'nn_training_loss_history1':training_loss_history1,
            'nn_validation_loss_history1':validation_loss_history1,
            'nn_model2': nn_model2.state_dict(),
            'nn_training_loss_history2':training_loss_history2,
            'nn_validation_loss_history2':validation_loss_history2,
            'in_features':in_features,
            'out_features':out_features,
            'hidden_layers':hidden_layers
            }
            

def TrainKanNetwork(in_features:int,out_features:int,hidden_layers:int):
    """Train the KAN Network
    
    https://github.com/Blealtan/efficient-kan/tree/master
    
    Args:
        in_features (int): Number of inputs
        out_features (int): Number of outputs 
        hidden_layers (int): Size of hidden layers. Example, [8,8,8,8] 4 layers with 8 neurons per layer 

    Returns:
        Dict: Dictionary containing the model states and training and validation loss history
        
    """
    kan_layers = [in_features]; kan_layers.extend(hidden_layers); kan_layers.append(out_features)
    kan_model1 = KAN(kan_layers)
    kan_model2 = KAN(kan_layers)
    
    model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader = CreateDataLoaders(in_features)
    
    kan_model1,training_loss_history1,validation_loss_history1 = train_model(kan_model1,model1_train_loader,model1_test_loader,num_epochs=2000,verbose=True)
    kan_model2,training_loss_history2,validation_loss_history2 = train_model(kan_model2,model2_train_loader,model2_test_loader,num_epochs=2000,verbose=True)  
    
    
    return {'KAN_model1': kan_model1.state_dict(),
            'KAN_training_loss_history1':training_loss_history1,
            'KAN_validation_loss_history1':validation_loss_history1,
            'KAN_model2': kan_model2.state_dict(),
            'KAN_training_loss_history2':training_loss_history2,
            'KAN_validation_loss_history2':validation_loss_history2
            }
            
    

def Step2_TrainModels():
    """Build the models and train them 
    """
    in_features = 5; out_features = 2 # For both model 1 and 2
    hidden_layers = [64,64,64,64]
    nn_data = TrainMultiLayer(in_features,out_features,hidden_layers)
    kan_data = TrainKanNetwork(in_features,out_features,hidden_layers)

    pickle.dump(dict(nn_data, **kan_data),
                open('checkpoints/02_5HoleProbe_Trained_Models.pkl','wb'))
    
    
if __name__ == "__main__":
    Step1_ProcessData()
    Step2_TrainModels()