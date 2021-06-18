import math

class SamplingScheduler:
    def __init__(self, max_epoch, init_val=1):
        self.val = init_val
        self.max_epoch = max_epoch

    def update_by_epoch(self, epoch):
        self.val =  self.val
    
    def get_val(self):
        return self.val 

class LinearScheduler(SamplingScheduler):
    def __init__(self, max_epoch, init_val=1):
        super().__init__(max_epoch, init_val=init_val)
    
    def update_by_epoch(self, epoch):
        self.val = 1 - epoch / self.max_epoch

class ConvexScheduler(SamplingScheduler):
    def __init__(self, max_epoch, init_val=1):
        super().__init__(max_epoch, init_val=init_val)
    
    def update_by_epoch(self, epoch):
        self.val = math.cos(epoch/self.max_epoch * math.pi/2)

class ConcaveScheduler(SamplingScheduler):
    def __init__(self, max_epoch, init_val=1, lambda_val = 0.99):
        self.lambda_val = lambda_val
        self.val = init_val
        self.max_epoch = max_epoch

    def update_by_epoch(self, epoch):
        self.val = self.lambda_val**epoch

class CompositeScheduler(SamplingScheduler):
    def __init__(self, max_epoch, init_val=1):
        super().__init__(max_epoch, init_val=init_val)
    
    def update_by_epoch(self, epoch):
        self.val = 0.5*math.cos(epoch/self.max_epoch * math.pi) + 0.5

class LossScheduler(SamplingScheduler):
    def __init__(self, max_epoch, init_val=1, p=0.3, epsilon=0.1):
        super().__init__(max_epoch, init_val=init_val+epsilon)
        
        self.epsilon = epsilon
        self.p = p

    def update_by_epoch(self, epoch):
        if epoch < self.max_epoch * self.p:
            self.val = 0.5*math.cos(epoch/self.max_epoch * math.pi) + 0.5 + self.epsilon
        else:
            self.val = self.epsilon
