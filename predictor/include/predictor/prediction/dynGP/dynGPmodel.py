import torch 
import array
import copy
import sys
import time
import gpytorch
from typing import Type, List
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import secrets
from barcgp.prediction.abstract_gp_controller import GPController
from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *
from gpytorch.mlls import SumMarginalLogLikelihood
from barcgp.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate

from barcgp.prediction.dynGP.dynGPdataGen import SampleGeneartorDynGP

class DynGPApproximate(GPController):
    def __init__(self, sample_generator: SampleGeneartorDynGP, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood, input_size: int, output_size: int, inducing_points: int,
                 enable_GPU=False):
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        self.model = IndependentMultitaskGPModelApproximate(inducing_points_num=inducing_points,
                                                            input_dim=self.input_size,
                                                            num_tasks=self.output_size)  # Independent
        self.independent = True        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup_dataloaders(self,train_dataload,valid_dataload, test_dataloader):
        self.train_loader = train_dataload
        self.valid_loader = valid_dataload
        self.test_loader = test_dataloader


    def pull_samples(self, holdout=150):        
        
        self.train_x = torch.zeros((self.sample_generator.getNumSamples() - holdout, self.input_size))  # [ego_state | tv_state]
        self.test_x = torch.zeros((holdout, self.input_size))  # [ego_state | tv_state]
        self.train_y = torch.zeros([self.sample_generator.getNumSamples() - holdout, self.output_size])  # [tv_actuation]
        self.test_y = torch.zeros([holdout, self.output_size])  # [tv_actuation]

        # Sampling should be done on CPU
        self.train_x = self.train_x.cpu()
        self.test_x = self.test_x.cpu()
        self.train_y = self.train_y.cpu()
        self.test_y = self.test_y.cpu()

        not_done = True
        sample_idx = 0
        while not_done:            
            samp = self.sample_generator.nextSample()
            if samp is not None:                
                samp_input, samp_output = samp
                if sample_idx < holdout:
                    self.test_x[sample_idx] = samp_input
                    self.test_y[sample_idx] = samp_output
                else:
                    self.train_x[sample_idx - holdout] = samp_input
                    self.train_y[sample_idx - holdout] = samp_output
                sample_idx += 1
            else:
                print('Finished')
                not_done = False        
      
        self.means_x = self.train_x.mean(dim=0, keepdim=True)
        self.stds_x = self.train_x.std(dim=0, keepdim=True)
        self.means_y = self.train_y.mean(dim=0, keepdim=True)
        self.stds_y = self.train_y.std(dim=0, keepdim=True)
        
        self.normalize = True
        if self.normalize:
            for i in range(self.stds_x.shape[1]):
                if self.stds_x[0, i] == 0:
                    self.stds_x[0, i] = 1
            self.train_x = (self.train_x - self.means_x) / self.stds_x
            self.test_x = (self.test_x - self.means_x) / self.stds_x

            for i in range(self.stds_y.shape[1]):
                if self.stds_y[0, i] == 0:
                    self.stds_y[0, i] = 1
            self.train_y = (self.train_y - self.means_y) / self.stds_y
            self.test_y = (self.test_y - self.means_y) / self.stds_y
            print(f"train_x shape: {self.train_x.shape}")
            print(f"train_y shape: {self.train_y.shape}")


    def outputToReal(self, output):
        if self.normalize:
            return output

        if self.means_y is not None:
            return output * self.stds_y + self.means_y
        else:
            return output

    def train(self):
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
            self.test_x = self.test_x.cuda()
            self.test_y = self.test_y.cuda()

        # Find optimal model hyper-parameters
        self.model.train()
        self.likelihood.train()

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.train_x), torch.tensor(self.train_y)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.test_x), torch.tensor(self.test_y)
        )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=150 if self.enable_GPU else 100,
                                      shuffle=True,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=25,
                                      shuffle=False,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)

        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.005)  # Includes GaussianLikelihood parameters

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_y.numel())

        epochs = 100
        last_loss = np.inf
        no_progress_epoch = 0
        not_done = True
        epoch = 0
        best_model = None
        best_likeli = None
        sys.setrecursionlimit(100000)
        while not_done:
        # for _ in range(epochs):
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0
            for step, (train_x, train_y) in enumerate(train_dataloader):
                # Within each iteration, we will go over each minibatch of data
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                train_loss += loss.item()
                train_dataloader.set_postfix(log={'train_loss': f'{(train_loss / (step + 1)):.5f}'})
                loss.backward()
                optimizer.step()
            for step, (train_x, train_y) in enumerate(valid_dataloader):
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})
            if c_loss > last_loss:
                if no_progress_epoch >= 15:
                    not_done = False
            else:
                best_model = copy.copy(self.model)
                best_likeli = copy.copy(self.likelihood)
                last_loss = c_loss
                no_progress_epoch = 0

            no_progress_epoch += 1
        self.model = best_model
        self.likelihood = best_likeli
    
    def evaluate(self):       
        self.set_evaluation_mode()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # This contains predictions for both outcomes as a list
            predictions = self.likelihood(self.likelihood(self.model(self.test_x[:50])))

        mean = predictions.mean.cpu()
        variance = predictions.variance.cpu()
        self.means_x = self.means_x.cpu()
        self.means_y = self.means_y.cpu()
        self.stds_x = self.stds_x.cpu()
        self.stds_y = self.stds_y.cpu()
        self.test_y = self.test_y.cpu()

        f, ax = plt.subplots(self.output_size, 1, figsize=(15, 10))
        titles = ['del_vlon', 'del_vlat', 'del_wpsi']
        for i in range(self.output_size):
            unnormalized_mean = self.stds_y[0, i] * mean[:, i] + self.means_y[0, i]
            unnormalized_mean = unnormalized_mean.detach().numpy()
            cov = np.sqrt((variance[:, i] * (self.stds_y[0, i] ** 2)))
            cov = cov.detach().numpy()
            '''lower, upper = prediction.confidence_region()
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()'''
            lower = unnormalized_mean - 2 * cov
            upper = unnormalized_mean + 2 * cov
            tr_y = self.stds_y[0, i] * self.test_y[:50, i] + self.means_y[0, i]
            # Plot training data as black stars
            ax[i].plot(tr_y, 'k*')
            # Predictive mean as blue line
            # ax[i].scatter(np.arange(len(unnormalized_mean)), unnormalized_mean)
            ax[i].errorbar(np.arange(len(unnormalized_mean)), unnormalized_mean, yerr=cov, fmt="o", markersize=4, capsize=8)
            # Shade in confidence
            # ax[i].fill_between(np.arange(len(unnormalized_mean)), lower, upper, alpha=0.5)
            ax[i].legend(['Observed Data', 'Predicted Data'])
            ax[i].set_title(titles[i])
        plt.show()


class DynGPApproximateTrained(GPController):
    def __init__(self, name, enable_GPU, model=None):
        if model is not None:
            self.load_model_from_object(model)
        else:
            self.load_model(name)
        self.enable_GPU = enable_GPU
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.means_x = self.means_x.cuda()
            self.means_y = self.means_y.cuda()
            self.stds_x = self.stds_x.cuda()
            self.stds_y = self.stds_y.cuda()
        else:
            self.model.cpu()
            self.likelihood.cpu()
            self.means_x = self.means_x.cpu()
            self.means_y = self.means_y.cpu()
            self.stds_x = self.stds_x.cpu()
            self.stds_y = self.stds_y.cpu()