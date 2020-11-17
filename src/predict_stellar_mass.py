from src.models.models import *
#from src.data_processing import gal_dataloader as dataloader
import dataloader as dataloader # for now 
#from src.utils.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser, SlurmCluster
from argparse import ArgumentParser, Namespace
import os
import random
import sys
import numpy as np
import pandas as pd
#from validation_Pk import generate_valid_Pk
import Pk_library as PKL
import matplotlib.pyplot as plt

class Galaxy_Model(pl.LightningModule):
    def __init__(self, hparams=None):
        super(Galaxy_Model,self).__init__()
        self.__check_hparams(hparams)
        self.hparams = hparams
        self.prepare_data()
        self._model = DM2Gal(BasicBlock,in_ch=2,ch=self.n_ch,nblocks=2)
        
    def forward(self,x):

        x = self._model(x) # returns the predicted mass of the galaxy at the center of this cube.

        return x  
    

    def _run_step(self, batch, batch_idx,step_name):

        maps, target_mass = batch
        predicted_mass = self(maps)
        x1 = maps[:,0][0].mean(axis=1)
        x2 = maps[:,1][0].mean(axis=1)


        if batch_idx % 5000 == 0:
            self._log_dist(x1,x2,target_mass, predicted_mass, step_name)  
        #smooth_l1_loss or mse_loss
        if self.loss_fn == 'mse':
            loss = F.mse_loss(predicted_mass,target_mass) #F.mse_loss also an option.. 

        elif self.loss_fn == 'huber':
            loss = F.smooth_l1_loss(predicted_mass,target_mass)
        else: raise Exception("Pick a loss function!")

        return loss
    
    def training_step(self, batch, batch_idx):
        train_loss = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        val_loss = self._run_step(batch, batch_idx, step_name='valid')
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_tensorboard_logs = {'avg_val_loss': avg_val_loss}

        processed_path = root  = '../dat/processed/'
        #valid_Pk = generate_valid_Pk(self,self.valid_loader)
        valid_coords = pd.read_csv(processed_path + 'valid_coords.csv')
        val_files = valid_coords['Coordinates'].values #in test or validation mode, use everything to recreate our mini universe.

        coords_cube = valid_coords[['x','y','z']]
        max_coord = coords_cube.to_numpy().max()
        min_coord = coords_cube.to_numpy().min()
        cube_len = int(max_coord - min_coord + 1)
        coords_cube = coords_cube - min_coord
        params_mean, params_std = 8.750728, 0.60161114
        cube_pred = np.zeros(shape = (cube_len,cube_len,cube_len)) 
        cube_target = np.zeros(shape = (cube_len,cube_len,cube_len)) 
        ParamsPred = []
        ParamsTrue = []
        for i, maps_params in enumerate(self.valid_loader): 
            maps = maps_params[0]
            params = maps_params[1]

            predicted_mass = self(maps.cuda()).detach().item()
            predicted_mass = predicted_mass * params_std + params_mean
            ParamsPred.append(predicted_mass)
            predicted_mass = (10**predicted_mass) - 1

            x = int(coords_cube['x'].values[i])
            y = int(coords_cube['y'].values[i])
            z = int(coords_cube['z'].values[i])
            cube_pred[x,y,z] = predicted_mass

            ParamsTrue.append(params_std*params[0] + params_mean)
            target_mass = (10**(params_std*params[0] + params_mean)) - 1
            cube_target[x,y,z] = target_mass

        cube_pred = cube_pred.astype(np.float32)
        cube_target = cube_target.astype(np.float32)
        BoxSize = (cube_len / 2048) * 75 #Mpc/h 

        MAS = None
        threads = 8
        axis = 0
                
        Pk_target = PKL.Pk(cube_target, BoxSize, axis, MAS, threads)
        k       = Pk_target.k3D
        Pk0_true     = Pk_target.Pk[:,0] #monopole

        
            
        Pk_pred = PKL.Pk(cube_pred, BoxSize, axis, MAS, threads)
        k       = Pk_pred.k3D
        Pk0_pred     = Pk_pred.Pk[:,0] #monopole

        fig1, ax1 = plt.subplots()
        ax1.loglog(k,Pk0_pred,label='Pred')
        ax1.loglog(k,Pk0_true,label="True")
        ax1.legend()
        ax1.set_ylabel("Pk")
        ax1.set_xlabel("k")

        tag1 = "validation Pk"
        self.logger.experiment.add_figure(tag1, fig1, global_step=self.trainer.global_step, close=True, walltime=None)

        fig2, ax2 = plt.subplots()
        ax2.hist(ParamsPred,label='Pred')
        ax2.hist(ParamsTrue,alpha=.75,label="True")
        ax2.legend()
        ax2.set_ylabel("freq")
        ax2.set_xlabel("log Msun")
        tag2 = "validation pdf"
        self.logger.experiment.add_figure(tag2, fig2, global_step=self.trainer.global_step, close=True, walltime=None)
        return {'val_loss': avg_val_loss, 'log': val_tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.learning_rate ,weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 5)
        return [optimizer],[scheduler] 
        
    def prepare_data(self):
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        self.train_loader, self.valid_loader = dataloader.create_datasets(self.realizations,self.batch_size, self.seed,self.weighted_trainer)
   
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        pass
        #return self.test_loader

    def _log_dist(self,x1,x2, y, y_hat, step_name, limit=1):
        x1 = x1.resize(1,65,65)
        x2 = x2.resize(1,65,65)
        self.logger.experiment.add_image(f'{step_name}_input_nbody', x1, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_input_subhalo', x2, self.trainer.global_step)

        self.logger.experiment.add_histogram(f'{step_name}_predicted_mass', y_hat, self.trainer.global_step)
        self.logger.experiment.add_histogram(f'{step_name}_target_mass', y, self.trainer.global_step)

    def __check_hparams(self, hparams):
        self.learning_rate = hparams.learning_rate if hasattr(hparams, 'learning_rate') else 0.001
        self.weight_decay = hparams.weight_decay if hasattr(hparams, 'weight_decay') else 0.
        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 64
        self.realizations = hparams.realizations if hasattr(hparams, 'realizations') else 4096
        self.seed = hparams.seed if hasattr(hparams, 'seed') else 32
        self._eval = hparams._eval if hasattr(hparams, '_eval') else False
        self.n_ch = hparams.n_ch if hasattr(hparams, 'n_ch') else 16
        self.testing = hparams.testing if hasattr(hparams, 'testing') else False
        self.weighted_trainer = hparams.weighted_trainer if hasattr(hparams,'weighted_trainer') else False
        self.loss_fn = hparams.loss_fn if hasattr(hparams,'loss_fn') else 'mse'
        self.best_model_path = hparams.best_model_path if hasattr(hparams,'best_model_path') else 'None'

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)
        #tunable parameters
        parser.opt_list('--learning_rate', type=float, default=0.000010000, options=[.0001,.00001], tunable=False)
        parser.opt_list('--weight_decay', type = float,default = 0., options = [0.005,0.01,0.00001,0],tunable = True)
        parser.opt_list('--n_ch', type = int, default = 128, options = [64,128],tunable = False)
        parser.opt_list('--batch_size', type=int, default=32, options = [16,128],tunable = False)
        # fixed parameters
        parser.add_argument('--realizations', type=int, default=40000) #
        parser.add_argument('--seed', type=int, default = 24)
        parser.add_argument('--weighted_trainer',type=bool, default = True)
        parser.add_argument('--loss_fn',type=str, default = 'mse') #which loss function to train with. 
        parser.add_argument('--best_model_path',type=str, default = 'None') #whether or not to start training with a previous best model. 
        return parser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Central_Model.add_model_specific_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
