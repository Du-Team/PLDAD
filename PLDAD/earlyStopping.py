import torch
import numpy as np
import os

class EarlyStopping:

    def __init__(self, opt, patience=7, verbose=False, delta=0, model_path=''):

        self.opt = opt
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_path = model_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Earlystopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.model_path)
