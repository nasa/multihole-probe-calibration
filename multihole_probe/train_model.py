import numpy as np
import torch,  copy
import torch.nn as nn
import torch.utils.data.sampler as sampler
import scipy
from dataclasses import dataclass
from typing import List
import copy

def TestTrainSplit(dataset_size,validation_split,shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed=20
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices,val_indices

@dataclass
class LayerChanges:
    LayerIndx: int
    ConditionNumber: float
    neurons_to_keep: List[int]
    neurons_to_remove: List[int]


def train_model(model:torch.nn.Module,train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader, 
                num_epochs:int,
                verbose:bool=False,
                initial_lr:float=0.005,
                use_lr_scheduler:bool=True):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model.cuda()

    validation_loss_history = np.zeros((num_epochs,1))
    training_loss_history = np.zeros((num_epochs,1))
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.9,min_lr=0.0001)
    # Train the model
    j=0
    for epoch in range(num_epochs):
        training_loss_avg = 0
        training_loss_min = 100
        training_loss_max = 0
        validation_loss_avg = 0
        i = 0
        npoints = train_loader.sampler.__len__()
        for local_batch,local_labels in train_loader: 
            # Move tensors to the configured device 
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(local_batch)
            loss = criterion(outputs,local_labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            training_loss_avg += 1/(i+1) * (loss.item()-training_loss_avg) # Moving Average
            if (loss.item()<training_loss_min):
                training_loss_min = loss.item()
            if (loss.item()>training_loss_max):
                training_loss_max = loss.item()

            j+=1
            if (i+train_loader.batch_size)<npoints:
                i+=1
            else:
                i+=(npoints-i*train_loader.batch_size)/train_loader.batch_size
        training_loss_history[epoch] = training_loss_avg
        with torch.set_grad_enabled(False):
            i = 0
            for local_batch,local_labels in val_loader: 
                # Move tensors to the configured device 
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                outputs = model(local_batch)
                validation_loss = criterion(outputs,local_labels)
                if use_lr_scheduler:
                    scheduler.step(validation_loss)
                validation_loss_avg += 1/(i+1) * (validation_loss.item()-validation_loss_avg) # Moving Average
                if (i+val_loader.batch_size)<npoints:
                    i+=1
                else:
                    i+=(npoints-i*val_loader.batch_size)/val_loader.batch_size
            validation_loss_history[epoch] = validation_loss_avg
        if (verbose):
            print('Epoch [{}/{}] Train Loss: {:.2e}, Validation Loss {:.2e}, Min Loss {:.2e}, Max Loss {:.2e}'
                .format(epoch+1, num_epochs, training_loss_avg, validation_loss_avg,training_loss_min,training_loss_max))

    return model,training_loss_history,validation_loss_history
