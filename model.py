### Functions for defining the network of a classifier
### Configured according to https://www.mdpi.com/2076-3417/13/8/4793
import torch
import torch.nn as nn


class TSC_M(nn.Module):
    def __init__(self,drop_prob=0.3) -> None:
        super(TSC_M, self).__init__()
        
        #grayscale input
        #(bs,1,32,32)
        self.c1 = nn.Sequential( nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU()    
                                )
        #(bs,32,28,28)
        self.mp_dp1= nn.Sequential( nn.MaxPool2d(2)#,
                                   #nn.Dropout(p=drop_prob)
                                   )
        #(bs,32,14,14)
        self.c2 = nn.Sequential( nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU()    
                                )
        #(bs,64,10,10)
        self.mp_dp2= nn.Sequential( nn.MaxPool2d(2)#,
                                   #nn.Dropout(p=drop_prob)
                                   )
        #(bs,64,5,5)
        ## a flatten should be applied to turn (bs,64,5,5) to (bs,64*5*5)
        self.dense1 = nn.Sequential( nn.Linear(in_features=64*5*5,out_features=512),
                                    nn.ReLU()#,
                                    #nn.Dropout(p=drop_prob)
        )
        #(bs,512)
        self.dense2 = nn.Sequential(nn.Linear(in_features=512,out_features=43),
                                    nn.Softmax(dim=1))
        #(bs,43)
    
    def forward(self,x:torch.Tensor):
        #print(f'what ? {x.shape}')
        #raise ValueError("stop")
        x = self.c1(x)
        #print(f'after c1: {x.shape}')
        x = self.mp_dp1(x)
        #print(f'after mpdp1: {x.shape}')
        x = self.c2(x)
        #print(f'after c2: {x.shape}')
        x = self.mp_dp2(x)
        #print(f'after mp_dp2: {x.shape}')
        x = torch.flatten(x,1)
        #print(f'after flattern: {x.shape}')
        x = self.dense1(x)
        #print(f'after dense1: {x.shape}')
        x = self.dense2(x)
        #print(f'after dense2: {x.shape}')
        #raise ValueError("wtf")
        return x

class TSC_MA(nn.Module):
    pass

