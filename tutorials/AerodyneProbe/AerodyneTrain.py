import sys, os, re, glob, pathlib
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
    FAP1 = df['P1[Torr]'].values.astype(float)	# Hole 5, Center Hole 
    FAP2 = df['P2[Torr]'].values.astype(float)	# Hole 1
    FAP3 = df['P3[Torr]'].values.astype(float)    # Hole 2
    FAP4 = df['P4[Torr]'].values.astype(float)    # Hole 3
    FAP5 = df['P5[Torr]'].values.astype(float)    # Hole 4
    Pt = df['Pt[Torr]'].values.astype(float)
    Ps = df['Ps[Abs,Torr]'].values.astype(float)
    # Predicted 
    theta = df['Theta[deg]'].values.astype(float) 	# Model 1
    phi = df['Phi[deg]'].values.astype(float) 	# Model 1
    mach = df['Mach'].values.astype(float)		# Model 2
    Cpt = (FAP1-Ps) / (Pt-Ps) 	# Model 2
    # Used to train model to predicted values of Theta, Phi, Mach, Cpt
    Pavg = 0.25 * (FAP2 + FAP3 + FAP4 + FAP5) # Average of flow angle ports
    Cp1 = (FAP2-Pavg)/(FAP1-Pavg) 
    Cp2 = (FAP3-Pavg)/(FAP1-Pavg)
    Cp3 = (FAP4-Pavg)/(FAP1-Pavg)
    Cp4 = (FAP5-Pavg)/(FAP1-Pavg)
    Cpm = (FAP1-Pavg)/(FAP1)
 
    data = np.stack([Cp1,Cp2,Cp3,Cp4,Cpm,theta,phi]).transpose()
    model1_data = pd.DataFrame(data=data, index=np.arange(0,len(Cp1)), columns=['Cp1','Cp2','Cp3','Cp4','Cpm','theta','phi'])
    
    data = np.stack([Cp1,Cp2,Cp3,Cp4,Cpm,mach,Cpt]).transpose()
    model2_data = pd.DataFrame(data=data, index=np.arange(0,len(Cp1)), columns=['Cp1','Cp2','Cp3','Cp4','Cpm','mach','Cpt'])

    return model1_data, model2_data

def Step1_ProcessData():
    """Process the data into test and train for both models
    """
    def read_csv(filename:str,start_index:int):
        df = None
        with open(filename,'r') as f:
            data = []; line = ""
            while not line.startswith("Theta[deg]"):
                line = f.readline().strip()
                
            header = line.strip().split('\t')
            for line in f:
                data.append(line.strip().split('\t'))
            df = pd.DataFrame(data, columns=header,index=np.arange(start_index,start_index+len(data)))
        return df
            
    def list_files_walk(start_path='.',ext='.csv'): # Find all 
        probelist = list()
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if file.endswith(ext):
                    probelist.append(root)
                    break
                    # fileList.append(os.path.join(root, file))
        return probelist
    
    os.makedirs('checkpoints',exist_ok=True)
    os.makedirs('results',exist_ok=True)
    
    probes_to_calibrate = list_files_walk('../../dataset/5Hole-Probe2 Aeroprobe/')
    probe_list = list(); 
    for probe_dir in probes_to_calibrate:
        start_index = 0
        datafiles = glob.glob(probe_dir+'/*.csv')
        probe_data = list()
        path = pathlib.Path(probe_dir)
        for m in datafiles:
            match = re.findall('M\d+',m)[0]
            mach = int(match[1:])/1000
            df_temp = read_csv(m,start_index)
            df_temp['Mach']=mach
            probe_data.append(df_temp)
            start_index += len(df_temp)
        df = pd.concat(probe_data,axis=0)
        df = df.fillna(0)
        
        model1_data,model2_data = CreateCoefficients(df)
        
        minmax_model1 = MinMaxScaler(feature_range=(-1, 1))
        minmax_model2 = MinMaxScaler(feature_range=(-1, 1))
        minmax_model1.fit(model1_data)
        minmax_model2.fit(model2_data)
        model1_data = minmax_model1.transform(model1_data)
        model2_data = minmax_model2.transform(model2_data)
        
        train1,test1 = train_test_split(model1_data,test_size=0.2)
        train2,test2 = train_test_split(model2_data,test_size=0.2)

        
        probe_list.append({
                        'name':path.name,
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
                            },
                        'dataset':df
                    })
    
    pickle.dump(probe_list,
                open('checkpoints/01_5HoleProbe_data.pkl','wb'))

def CreateDataLoaders(in_features:int,
                      train_data1:pd.DataFrame,
                      test_data1:pd.DataFrame,
                      train_data2:pd.DataFrame,
                      test_data2:pd.DataFrame):
    """Read in the data and create the torch data loaders 

    Args:
        in_features (int): number of input features 
        train_data1 (pd.DataFrame): Training data for model 1 
        test_data1 (pd.DataFrame): Test Data for model 1
        train_data2 (pd.DataFrame): Training data for model 2 
        test_data2 (pd.DataFrame): Test Data for model 2
        
    Returns:
        (Tuple) containing:
        
            *model1_train_loader* (DataLoader): DataLoader to predict Theta and Phi
            *model1_test_loader* (DataLoader): DataLoader to validate Theta and Phi
            *model2_train_loader* (DataLoader): DataLoader to predict Mach and Cpt
            *model2_test_loader* (DataLoader): DataLoader to validate Mach and Cpt
            
    """
    
    # Multi-layer perception
    model1_x_train = train_data1[:,:in_features] 
    model1_y_train = train_data1[:,in_features:]
    model1_x_test = test_data1[:,:in_features] 
    model1_y_test = test_data1[:,in_features:]
    
    model2_x_train = train_data2[:,:in_features] 
    model2_y_train = train_data2[:,in_features:]
    model2_x_test = test_data2[:,:in_features] 
    model2_y_test = test_data2[:,in_features:]
    
    model1_train_dataset = torch.utils.data.TensorDataset(torch.tensor(model1_x_train).float(),torch.tensor(model1_y_train).float())
    model1_test_dataset = torch.utils.data.TensorDataset(torch.tensor(model1_x_test).float(),torch.tensor(model1_y_test).float())
    
    model2_train_dataset = torch.utils.data.TensorDataset(torch.tensor(model2_x_train).float(),torch.tensor(model2_y_train).float())
    model2_test_dataset = torch.utils.data.TensorDataset(torch.tensor(model2_x_test).float(),torch.tensor(model2_y_test).float())
    
    model1_train_loader = torch.utils.data.DataLoader(dataset=model1_train_dataset,batch_size=64)
    model1_test_loader = torch.utils.data.DataLoader(dataset=model1_test_dataset,batch_size=64)
    
    model2_train_loader = torch.utils.data.DataLoader(dataset=model2_train_dataset,batch_size=64)
    model2_test_loader = torch.utils.data.DataLoader(dataset=model2_test_dataset,batch_size=64)
    return model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader
    
def TrainMultiLayer(in_features:int,out_features:int,hidden_layers:int,
                    train_data1:pd.DataFrame,test_data1:pd.DataFrame,
                    train_data2:pd.DataFrame,test_data2:pd.DataFrame):
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
    
    model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader = CreateDataLoaders(in_features,train_data1,test_data1,train_data2,test_data2)
    
    nn_model1,training_loss_history1,validation_loss_history1 = train_model(nn_model1,model1_train_loader,model1_test_loader,
                                                                            num_epochs=2000,
                                                                            verbose=True,
                                                                            initial_lr=0.01,
                                                                            use_lr_scheduler=True)
    
    nn_model2,training_loss_history2,validation_loss_history2 = train_model(nn_model2,model2_train_loader,model2_test_loader,
                                                                            num_epochs=2000,
                                                                            verbose=True,
                                                                            initial_lr=0.01,
                                                                            use_lr_scheduler=True)
    
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
            

def TrainKanNetwork(in_features:int,out_features:int,hidden_layers:int,        
                    train_data1:pd.DataFrame,test_data1:pd.DataFrame,
                    train_data2:pd.DataFrame,test_data2:pd.DataFrame):
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
    
    model1_train_loader, model1_test_loader, model2_train_loader, model2_test_loader = CreateDataLoaders(in_features,train_data1,test_data1,train_data2,test_data2)
    
    kan_model1,training_loss_history1,validation_loss_history1 = train_model(kan_model1,model1_train_loader,model1_test_loader,num_epochs=2000,verbose=True,
                                                                            initial_lr=0.001,
                                                                            use_lr_scheduler=False)
    kan_model2,training_loss_history2,validation_loss_history2 = train_model(kan_model2,model2_train_loader,model2_test_loader,num_epochs=2000,verbose=True,
                                                                            initial_lr=0.001,
                                                                            use_lr_scheduler=False)  
    
    
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
    hidden_layers = [128,128,128,64]
    data = pickle.load(open('checkpoints/01_5HoleProbe_data.pkl', 'rb'))
    for m in data:
        name = m['name']
        if not os.path.exists(f'checkpoints/{name}_Trained_Models.pkl','wb'):
            nn_data = TrainMultiLayer(in_features,out_features,hidden_layers,
                                    m['model1']['train'],m['model1']['test'],
                                    m['model2']['train'],m['model2']['test'])
            kan_data = TrainKanNetwork(in_features,out_features,hidden_layers,
                                    m['model1']['train'],m['model1']['test'],
                                    m['model2']['train'],m['model2']['test'])
            
            pickle.dump(dict(nn_data, **kan_data),
                    open(f'checkpoints/{name}_Trained_Models.pkl','wb'))
    
    
if __name__ == "__main__":
    Step1_ProcessData()
    Step2_TrainModels()