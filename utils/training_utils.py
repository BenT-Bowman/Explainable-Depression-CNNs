import torch
import os
import numpy as np

class EarlyStop():
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name

        self.last_loss = np.inf
        self.since_last = 0

    def __call__(self, model, val_loss, patience):
        model_path = os.path.join(self.save_dir, self.model_name)
        
        if val_loss < self.last_loss:
            self.last_loss = val_loss
            torch.save(model.state_dict(), model_path)
            self.since_last = 0
        else:
            print(f"Hasn't improved in {self.since_last}")
            self.since_last += 1
        
        if self.since_last > patience:
            return True
        return False