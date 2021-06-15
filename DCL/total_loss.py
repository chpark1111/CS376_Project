import torch
import torch.nn as nn
import torch.nn.functional as F

from DCL.DSL_loss import DSL_loss
from DCL.TEA_loss import TEA_Loss
from DCL.scheduler import LossScheduler

class total_loss(nn.Module):
    def __init__(self, train_dataloader, max_epoch):
        '''
        Do not forget to init
        '''
        super().__init__()

        self.TEA_loss = TEA_Loss()
        self.DSL_loss = DSL_loss(train_dataloader, max_epoch)
        self.scheduler = LossScheduler(max_epoch)

    def forward(self, pred, gt):
        '''
        Use it like loss function
        '''
        return self.DSL_loss(pred, gt) + self.scheduler.get_val()*self.TEA_loss(pred, gt)
    
    def step(self, epoch):
        '''
        Call every start of epoch
        '''
        self.DSL_loss.step(epoch)
        self.scheduler.update_by_epoch(epoch)
