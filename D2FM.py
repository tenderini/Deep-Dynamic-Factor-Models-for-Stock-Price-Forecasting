## Barcelona School of Economics - Master in Data Science Methodology
## Deep Learning - Final Project
## ---------------------------------------------------------------------------
## Authors: Alessandro Ciancetta and Alessandro Tenderini
## ---------------------------------------------------------------------------

## Description
# This script contains the definition of all the relevant classes and functions for estimating a 
# Deep Dynamic Factor Model (Andreini, Izzo and Ricco 2020). 
# We extend the implementation proposed in the paper by using a GRU to account for the
# non-linear dynamics of the latent states. 
# This code is not based on any previous implementation of the model and is completely our own code.

## -----------------------------------------------------------------------------------------------------
## Contents
## -----------------------------------------------------------------------------------------------------
## 1) Definition of the Autoencoder class (and of the ancillary classes for data loading)
## 2) Definition of the GRU class (and of the ancillary classes for data loading)
## 3) Definition of the D2FM class, that combines the two previous models
## 4) Definition of the ForecastEvaluation class, for assessing the performances of the D2FM
## 5) Definition of the GridSearch class, for tuning the hyperparameters of the model
## -----------------------------------------------------------------------------------------------------


## Load relevant modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt


## =====================================================================================================
## 1) Definition of the autoencoder
## =====================================================================================================

## Dataset class to pass in the DataLoader for batch learning
class Data(Dataset):
    def __init__(self, train_df):
        self.X = torch.from_numpy(train_df)
        self.len = self.X.shape[0]
        self.nfeatures = self.X.shape[1]

    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return self.len
        
## make_loader function to build the loader for each training set in the sliding-window procedure
def make_loader(train_df, batch_size):
    data = Data(train_df)
    loader = DataLoader(dataset = data, batch_size=batch_size, shuffle=False) # time series data --> do not shuffle!
    return loader


## Flexible autoencoder, with variable depth
class Autoencoder(nn.Module):
    def __init__(self, n_features, hidden_layer_sizes, latent_size, decoder_type, device, load_params=True):
        super(Autoencoder, self).__init__()

        # Define the encoder layers
        encoder_layers = []
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                encoder_layers.append(nn.Linear(n_features, hidden_layer_sizes[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_layer_sizes[-1], latent_size))
        encoder_layers.append(nn.Tanh()) # to have (-1,1) latent variables
        self.encoder = nn.Sequential(*encoder_layers)

        # Define the decoder layers
        if decoder_type == "linear":
            self.decoder = torch.nn.Linear(latent_size, n_features) 

        elif decoder_type == "symmetric":
            decoder_layers = []
            decoder_layers.append(nn.Linear(latent_size, hidden_layer_sizes[-1]))
            decoder_layers.append(nn.ReLU())
            for i in range(len(hidden_layer_sizes)-1, 0, -1):
                decoder_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i-1]))
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(hidden_layer_sizes[0], n_features))
            self.decoder = nn.Sequential(*decoder_layers)
            
        else:
            raise ValueError("decoder_type must be either 'linear' or 'symmetric'")

        # Other elements in self
        self.device = device
        self.filename = "learned_parameters/AE_input"+str(n_features)+"_"+decoder_type+"_".join([str(hidden_layer_sizes[i]) for i in range(len(hidden_layer_sizes))])+ ".pt"
        # Move the model to the device
        self.to(device)
        # Load pre-trained parameters (if requested and available)
        if load_params:
            try:
                self.load_state_dict(torch.load(self.filename))
                print("Parameters of the autoencoder successfully loaded from", self.filename)
            except FileNotFoundError:
                pass
            except RuntimeError:
                print("Parameters already existing in a file but the dimensions are incompatible")
                pass

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, x, batch_size, epochs, lr=0.001, schedule=0.999):
        ## Optimization: Adam with MSE loss function. Learning rate schedule: exponential decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = ExponentialLR(optimizer, gamma=schedule)
        loader = make_loader(x, batch_size)
        for epoch in range(epochs):
            for x in loader:
                x = x.to(self.device)
                pred = self.forward(x.float())
                loss = criterion(pred, x.float()) #(output, target)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch+1)%10 == 0:
                print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
        
    def save(self):
        torch.save(self.state_dict(), self.filename)



## =====================================================================================================
## 2) Definition of the GRU
## =====================================================================================================

# Creating the dataset
## X = list of sequences of past data. Size of the tensor = (number of lags i.e. sequence length, number of total observations, number of variables),
## y = list of target values. Size of the tensor = (number of observations, size of the output i.e. number of variables to predict)
## s = sequence length
get_seq = lambda d, s=7: torch.Tensor(np.vstack([d[i:(s+i)] for i in range(len(d)-s)])).view(s, len(d)-s, -1)
get_tag = lambda d, s=7: torch.Tensor(np.vstack([ d[s+i] for i in range(len(d)-s)]))

# Create iterable Dataset to pass into DataLoader for batch learning
class DataSequences(Dataset):
    def __init__(self, train_df, seq_length=7):
        self.X = get_seq(train_df, s=seq_length)
        self.y = get_tag(train_df, s=seq_length)

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        ## get an observation from the sequences and labels: X[all lags, specific batch, all features].
        return self.X[:,idx,:], self.y[idx,:]


# Gated Recurrent Units
class GRU(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, device, load_params=True):
        super(GRU, self).__init__()

        # Model blocks (core)
        self.gru = nn.GRU(n_features, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, n_features)

        # Move the model to the device
        self.device = device
        self.to(device)
        # Load pre-trained parameters (if required and available)
        self.filename = "learned_parameters/GRU_input"+str(n_features)+"_hidden"+str(hidden_size)+"_layers"+str(num_layers)+".pt"
        if load_params:
            try:
                self.load_state_dict(torch.load(self.filename))
                print("Parameters of the GRU successfully loaded from", self.filename)
            except FileNotFoundError:
                pass
            except RuntimeError:
                print("Parameters already  existing in a file but the dimensions are incompatible")
                pass

    def forward(self, x):
        # output, (h_n, c_n) = self.gru(x)
        output, h_n = self.gru(x)
        output = output[-1, :, :] # get the output for the last time step
        output = self.fc(output)
        return output
    
    def fit(self, x, batch_size, epochs, lr = 0.001, schedule=0.999, seq_length=7):
        ## Optimization: Adam with MSE loss function. Learning rate schedule: exponential decay
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = ExponentialLR(optimizer, gamma=schedule)

        ## Get (labelled) sequences and batches of sequences
        train_data = DataSequences(x, seq_length=seq_length)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch_data, batch_labels) in train_loader:
                batch_data = batch_data.transpose(1,0).to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Forward pass
                output = self.forward(batch_data)
                # Compute the loss
                loss = criterion(output, batch_labels)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_size
            scheduler.step()
            if (epoch+1)%10==0:
                # print(f"Epoch {epoch+1} - Loss: {running_loss/batch_data.size(1):.4f}")
                print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    def save(self):
        torch.save(self.state_dict(), self.filename)



## =====================================================================================================
## 3) Definition of the D2FM model class
## =====================================================================================================

class D2FM():
    def __init__(
            self, 
            n_features, latent_size, 
            hidden_layer_sizes, decoder_type, 
            hidden_size_gru, num_layers_gru, seq_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        ## Non-linear encoding to the latent factors using the autoencoder
        self.model_static = Autoencoder(
                        n_features = n_features, 
                        hidden_layer_sizes=hidden_layer_sizes, 
                        latent_size=latent_size, 
                        decoder_type=decoder_type, 
                        device=device
                        )
        
        ## Non-linear dynamics between latent states using the GRU
        self.model_dynamic = GRU(
                        n_features=latent_size, 
                        hidden_size=hidden_size_gru, 
                        num_layers=num_layers_gru, 
                        device=device
                        )  
        self.seq_length = seq_length
        self.device = device

    def fit(self, d, batch_size, epochs, lr=0.001, schedule=0.999, autosave=True): 

        # Train static model
        print(f"----- Start fitting autoencoder -----")
        self.model_static.fit(d, batch_size = batch_size, epochs=epochs, schedule=schedule)

        # Get latent factors (i.e. get the training data for the GRU)
        d_tensor = torch.Tensor(d).to(self.device)
        self.static_factors = self.model_static.encoder(d_tensor).cpu().detach().numpy()

        # Train dynamic model
        print(f"----- Start fitting GRU -----")
        self.model_dynamic.fit(self.static_factors, batch_size = batch_size, epochs=epochs, schedule=schedule, seq_length=self.seq_length)

        # Save parameters in the "learned_parameters" folder (it must be an existing folder)
        if autosave:
            self.model_static.save()
            self.model_dynamic.save()
    
    def predict_next(self):
        # Predict naxt latent state
        get_seq = lambda d, s=7: torch.Tensor(np.vstack([d[i:(s+i)] for i in range(len(d)-s)])).view(s, len(d)-s, -1)
        last_sequence = get_seq(self.static_factors, s=self.seq_length)[:,-1,:].view(self.seq_length, 1, -1).to(self.device)
        predicted_factors = self.model_dynamic(last_sequence)
        # Reconstruct next predicted variables
        predicted_variables = self.model_static.decoder(predicted_factors).cpu().detach().numpy()
        return predicted_variables



## =====================================================================================================
## 4) Definition of the ForecastEvaluation class
## =====================================================================================================

## Evaluation of the forecasts produced by the D2FM using an expanding-window approach
class ForecastEvaluation():
    def __init__(self, d, hyper_params):
        ## Instantiate the D2FM
        self.model = D2FM(n_features=d.shape[1], latent_size=hyper_params["latent_size"], 
                    hidden_layer_sizes=hyper_params["hidden_layer_sizes"], 
                    decoder_type=hyper_params["decoder_type"], 
                    hidden_size_gru=hyper_params["hidden_size_gru"], 
                    num_layers_gru=hyper_params["num_layers_gru"], 
                    seq_length=hyper_params["seq_length"],
                    device=hyper_params["device"])
        
        self.hyper_params = hyper_params
        self.d = d

    def evaluate(self):
        ## Store test set
        idx_test = len(self.d) - self.hyper_params["test_size"]
        self.test_data = self.d[idx_test:,:]

        ## Expanding window forecast evaluation
        predictions = []
        for t in range(self.hyper_params["test_size"]):
            print(f"===== PREDICTION {t+1} of {self.hyper_params['test_size']} =====")
            ## Update the parameters
            idx_train = len(self.d) - self.hyper_params["test_size"] + t        
            self.model.fit(
                d=self.d[:idx_train,:], 
                batch_size=self.hyper_params["batch_size"], 
                epochs=self.hyper_params["epochs"], 
                lr=self.hyper_params["lr"], 
                schedule=self.hyper_params["schedule"], 
                autosave=self.hyper_params["autosave"])
            
            ## Predict next value
            predictions.append(self.model.predict_next())
        
        self.predictions = np.vstack(predictions)
        
    def get_predictions(self):
        return self.predictions
    
    def plot_predictions(self, series_index=1):
        plt.plot(self.test_data[:,series_index])
        plt.plot(self.predictions[:,series_index])
        plt.show()
        plt.plot(self.d[:,series_index])
        plt.plot(np.concatenate([np.zeros(len(self.d) - self.hyper_params["test_size"])*np.nan, self.predictions[:,series_index]]))
        plt.show()
    
    def get_rmse(self):
        return np.sqrt(np.mean((self.predictions-self.test_data)**2))



## =====================================================================================================
## 4) Definition of the GridSearch class
## =====================================================================================================

class GridSearch():
    def __init__(self, d):
        self.d = d

    def grid_search(
            self,
            latent_size_values = [4,8],
            hidden_layer_values = np.array([[64, 32, 16],[128, 64, 32, 16]], dtype=object), # at least two
            decoder_type_values = ["symmetric"],
            seq_length_values = np.array([7, 14, 21]),
            hidden_size_gru_values = [128],
            num_layers_gru_values = [2],
            schedule_values = [0.999],
            lr_values = [0.001],
            # Fixed parameters,
            test_size_fixed = [30],
            batch_size_fixed = [500],
            epochs_fixed = [50],
            autosave_fixed = False,
            device_fixed = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ):
        
        ## Initialize grid of parameters
        (
         latent_size_grid, hidden_layer_grid, decoder_type_grid, seq_length_grid, hidden_size_gru_grid, num_layers_gru_grid, schedule_grid, lr_grid, 
         test_size_grid, batch_size_grid, epochs_grid, autosave_grid, device_grid
        ) = np.meshgrid(
            latent_size_values, 
            hidden_layer_values,
            decoder_type_values, 
            seq_length_values, 
            hidden_size_gru_values, 
            num_layers_gru_values,
            schedule_values, 
            lr_values, 
            test_size_fixed, 
            batch_size_fixed, 
            epochs_fixed, 
            autosave_fixed, 
            device_fixed
            )

        param_grid = zip(latent_size_grid.flatten(), 
            hidden_layer_grid.flatten(), 
            decoder_type_grid.flatten(), 
            seq_length_grid.flatten(), 
            hidden_size_gru_grid.flatten(), 
            num_layers_gru_grid.flatten(), 
            schedule_grid.flatten(), 
            lr_grid.flatten(), 
            test_size_grid.flatten(), 
            batch_size_grid.flatten(), 
            epochs_grid.flatten(), 
            autosave_grid.flatten(), 
            device_grid.flatten()
         )

        hyper_param_keys = [
            'latent_size', 'hidden_layer_sizes', 'decoder_type',
            'seq_length', 'hidden_size_gru', 'num_layers_gru',
            'schedule', 'lr', 
            'test_size', 'batch_size', 'epochs', 'autosave', 'device'
            ]

        grid_list = []
        for p in param_grid:
            grid_list.append(dict(zip(hyper_param_keys, p)))
        
        self.grid_param = grid_list

        ## Evaluate the model on each point of the grid
        rmses = []
        rmse_best = 100
        for i, hyper_param in enumerate(grid_list):
            hyper_param["batch_size"] = int(hyper_param["batch_size"]) # otherwise it is recognized as float and an error arises
            print("\n--------------")
            print(f"Point of the grid {i+1} of {len(grid_list)}")
            print("--------------")

            ## Evaluate on the grid point using the ForecastEvaluation class
            forecast_evaluation = ForecastEvaluation(self.d, hyper_param)
            forecast_evaluation.evaluate()
            rmse = forecast_evaluation.get_rmse()
            rmses.append(rmse)
            if rmse < rmse_best:
                rmse_best = rmse
                with open('learned_parameters/log_param.csv','a') as fd:
                    fd.write("\n"+str(hyper_param))
        self.grid_rmse = rmses

    def get_best_param(self, verbose=False):
        if verbose:
            print(f"The RMSE of the best model is {np.min(self.grid_rmse)}")
        return self.grid_param[np.argmin(self.grid_rmse)]







