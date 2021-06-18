import torch
import torch.nn as nn
import torch.nn.functional as F

class TEA_Loss(nn.Module):
    def __init__(self, k=2, margin=1.25, minor_ratio = 0.4, easy_criterion=0.9):
        super(TEA_Loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='sum')
        self.k = k # Mine up to k number of hard positive, negative
        self.minor_ratio = minor_ratio #Ratio of how much percent of data we will use as minor data
        self.easy_criterion = easy_criterion #Definition of easy anchors ceriterion (0~1)

    def forward(self, pred, gt):
        '''
        pred: [batch, num_class]
        gt: [batch,]
        '''
        n = pred.shape[0] #batch_sz
        mask = gt.expand(n, n).eq(gt.expand(n, n).t()) #Whether i, j label is same

        minor = []
        minor_sample = []
        d_cnt = torch.bincount(gt)
        for i in range(len(d_cnt)):
            minor.append((d_cnt[i], i))
        minor.sort()
        sm = 0    
        for i in minor:
            if i[0]+sm <= n*self.minor_ratio:
                minor_sample.append(i[1])
                sm += i[0]
        
        AP_dis, AN_dis = [], []
        for i in range(n):
            if gt[i] not in minor_sample: #minor sample selection
                continue
            prob = F.softmax(pred[i], dim=-1)
            if prob[gt[i]] <= self.easy_criterion: #easy sample selection
                continue
            
            pos = pred[mask[i], gt[i]]
            pos.sort(descending=True)
            pos = pos[:self.k]
            neg = pred[(mask[i] == False), gt[i]]
            neg.sort()
            neg = neg[:self.k]

            for j in pos:
                for k in neg:
                    AP_dis.append((pred[i][gt[i]]-j).abs().unsqueeze(0))
                    AN_dis.append((pred[i][gt[i]]-k).unsqueeze(0))

        if len(AP_dis):
            AP_dis = torch.cat(AP_dis)
            AN_dis = torch.cat(AN_dis)
            y = torch.ones_like(AP_dis)
            loss = self.ranking_loss(AN_dis, AP_dis, y) / y.shape[0]
        else: loss = 0.0
        return loss 