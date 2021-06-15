import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DCL.scheduler import LinearScheduler, ConvexScheduler, ConcaveScheduler, CompositeScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Use big batch size
class DSL_loss(nn.Module):
    def __init__(self, train_dataloader, max_epoch, scheduler_type = 'convex'):
        super(DSL_loss, self).__init__()
        if scheduler_type == 'convex':
            self.scheduler = ConvexScheduler(max_epoch)
        elif scheduler_type == 'linear':
            self.scheduler = LinearScheduler(max_epoch)
        elif scheduler_type == 'concave':
            self.scheduler = ConcaveScheduler(max_epoch)
        elif scheduler_type == 'composite':
            self.scheduler = CompositeScheduler(max_epoch)
        else:
            print("Not supported scheduler type")
            assert 0
        
        self.d_train = dict()
        for image, label in train_dataloader:
            d_cnt = torch.bincount(label)
            for i in range(len(d_cnt)):
                if i not in self.d_train.keys():
                    self.d_train[i] = float(d_cnt[i])
                else:
                    self.d_train[i] += float(d_cnt[i])
        C_min = min(self.d_train.values())
        for i in self.d_train.keys():
            self.d_train[i] = self.d_train[i]/C_min       
        self.d_target = copy.deepcopy(self.d_train)

    def forward(self, pred, gt):
        '''
        pred: [batch, num_class]
        gt: [batch]
        Input is batched
        '''
        batch_sz = pred.shape[0]
        loss = 0.

        d_cnt = torch.bincount(gt)
        for i in range(len(d_cnt)):
            if not d_cnt[i]:
                d_cnt[i] = 1e5
        C_min = torch.min(d_cnt)
        for i in range(len(d_cnt)):
            if d_cnt[i]==1e5:
                d_cnt[i]= 0
        assert C_min!=0
        d_current = d_cnt / C_min
        pred = F.log_softmax(pred, dim=-1)

        for j in range(len(d_current)): #Think a way to replace this for loop
            if d_current[j]==0:
                continue
            if self.d_target[j]/d_current[j]<1:
                mask = (gt == j)
                sample = torch.zeros((pred[mask].shape[0],), dtype=torch.bool).to(device)
                smp_idx = np.random.choice(np.arange(pred[mask].shape[0]), int(pred[mask].shape[0]*self.d_target[j]/d_current[j]), replace=False) 
                smp_idx = torch.tensor(smp_idx).to(device)
                sample[smp_idx] = True
                gt_j = torch.tensor(j).expand(pred[mask][sample].shape[0]).to(device)
                loss += F.nll_loss(pred[mask][sample], gt_j, reduction = 'sum')
            else:    
                w = self.d_target[j]/d_current[j]
                mask = (gt == j)
                gt_j = torch.tensor(j).expand(pred[mask].shape[0]).to(device)
                loss += w*F.nll_loss(pred[mask], gt_j, reduction = 'sum')

        return loss/batch_sz
    
    def step(self, epoch):
        '''
        Call every start of epoch
        '''
        self.scheduler.update_by_epoch(epoch)
        for i in self.d_target.keys():
            self.d_target[i] = self.d_train[i]**self.scheduler.get_val()
