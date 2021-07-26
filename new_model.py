import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import models_2020


#model for training and validation



####################nora################"
#Nora ConvNet (deprecated)
class ConvNetComponent(nn.Module):
    """
    feature extraction from Nora's ConvNet
    (see CNN, https://arxiv.org/abs/1912.00852)
    (modified)
    """
    def __init__(self):
        super(ConvNetComponent, self).__init__()

        self.channel_sizes = [16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256]
        #self.channel_sizes = [16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256*2]
        kernel_size =7
#      kernel_size=5  
        self.features = nn.Sequential( 
            
            #16
            nn.BatchNorm1d(1),
            nn.Conv1d(1, self.channel_sizes[0], kernel_size=kernel_size),
            nn.BatchNorm1d(self.channel_sizes[0]),
            nn.ReLU(),
            #nn.Dropout(p=0.2))
        
            nn.Conv1d(self.channel_sizes[0], self.channel_sizes[1], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[1]),
            nn.ReLU(),
#            nn.Dropout(p=0.1),
            #nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[2]),
            nn.ReLU(),
 #           nn.Dropout(p=0.1),
        
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[2], self.channel_sizes[3], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[3]),
            nn.ReLU(),
#            nn.Dropout(p=0.2),
            #nn.AvgPool1d(5, stride=2),
            nn.Conv1d(self.channel_sizes[3], self.channel_sizes[4], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[4]),
            nn.ReLU(),
#            nn.Dropout(p=0.2),
            
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[4], self.channel_sizes[5], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[5]),
            nn.ReLU(),
 #           nn.Dropout(p=0.3),
            #nn.AvgPool1d(5, stride=2),
            nn.Conv1d(self.channel_sizes[5], self.channel_sizes[6], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[6]),
            nn.ReLU(),
 #           nn.Dropout(p=0.3),
                
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[6], self.channel_sizes[7], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[7]),
            nn.ReLU(),
#            nn.Dropout(p=0.4),
            #nn.AvgPool1d(5, stride=2),
            nn.Conv1d(self.channel_sizes[7], self.channel_sizes[8], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[8]),
            nn.ReLU(),
#            nn.Dropout(p=0.4),
            
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[8], self.channel_sizes[9], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[9]),
            nn.ReLU(),
#            nn.Dropout(p=0.5),
            #nn.AvgPool1d(5, stride=2),
            nn.Conv1d(self.channel_sizes[9], self.channel_sizes[10], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[10]),
            nn.ReLU(),
#            nn.Dropout(p=0.5),
            ###end for transformer
           #  #nn.AvgPool1d(2, stride=2),
#             nn.Conv1d(self.channel_sizes[10], self.channel_sizes[11], kernel_size=kernel_size), #,stride=(2,1)
#             nn.BatchNorm1d(self.channel_sizes[11]),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.AvgPool1d(5, stride=2),
#             nn.Conv1d(self.channel_sizes[11], self.channel_sizes[12], kernel_size=kernel_size), #,stride=(2,1)
#             nn.BatchNorm1d(self.channel_sizes[12]),
#             nn.ReLU(),
# #            nn.Dropout(p=0.5),
            
#             nn.AvgPool1d(2, stride=2), 
#             nn.Conv1d(self.channel_sizes[12], self.channel_sizes[13], kernel_size=kernel_size), #,stride=(2,1)
#             nn.BatchNorm1d(self.channel_sizes[13]),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             #nn.AvgPool1d(5, stride=2),
#             nn.Conv1d(self.channel_sizes[13], self.channel_sizes[14], kernel_size=kernel_size), #,stride=(2,1)
#             nn.BatchNorm1d(self.channel_sizes[14]),
#             nn.ReLU(),
            nn.Dropout(p=0.5)
            )
    
    
    def forward(self, inputs): 
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        #print(inputs.shape)
        output = self.features(inputs) 
        return output
    
class OldNora_ConvNet(nn.Module):

    def __init__(self,num_classes=26):
        super(OldNora_ConvNet, self).__init__()
        #9 classsifers
        num_features = 128
        num_mid = 128
        self.features = ConvNetComponent()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        #self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = nn.Sequential(nn.Linear(num_features, num_mid),nn.ReLU(),nn.Linear(num_mid,num_classes))

    def forward(self, inputs,src_mask=None):
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
        output = self.global_pool(output)
        #print(output.view(act_batch_size,-1).shape)
        output = self.classifier(output.view(act_batch_size, -1))
        return output
    
    def get_features(self,inputs):
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
        output = self.global_pool(output)
        return output

    
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)



#################SENet###################
#squeez and excitation block 
class SEInceptionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEInceptionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _= x.size()
        
        y = self.avg_pool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1)
        return x * y.expand_as(x)+x

class ConvNetSEComponent(nn.Module):
    """
    feature extraction from Nora's ConvNet
    (see CNN, https://arxiv.org/abs/1912.00852)
    modified by adding squeeze excitation layers
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self):
        super(ConvNetSEComponent, self).__init__()

        #self.channel_sizes = [16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128]
        self.channel_sizes = [16, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256,256,256,512,512,512]
        kernel_size =11
        self.features = nn.Sequential( 
            
            #16
            nn.BatchNorm1d(1),
            nn.Conv1d(1, self.channel_sizes[0], kernel_size=kernel_size),
            nn.BatchNorm1d(self.channel_sizes[0]),
            nn.ReLU(),

        
            nn.Conv1d(self.channel_sizes[0], self.channel_sizes[1], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[1]),
            nn.ReLU(),
           # nn.Dropout(p=0.5),
            
            nn.Conv1d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[2]),
            nn.ReLU(),
           # nn.Dropout(p=0.5),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(self.channel_sizes[2], self.channel_sizes[3], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[3]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[3]),
            
            nn.Conv1d(self.channel_sizes[3], self.channel_sizes[4], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[4]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[4]),
            
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[4], self.channel_sizes[5], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[5]),
            nn.ReLU(),
            #nn.Dropout(p=0.5)
            SEInceptionLayer(self.channel_sizes[5]),
            
            nn.Conv1d(self.channel_sizes[5], self.channel_sizes[6], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[6]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[6]),
                
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[6], self.channel_sizes[7], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[7]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[7]),
            
            nn.Conv1d(self.channel_sizes[7], self.channel_sizes[8], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[8]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[8]),


            nn.Conv1d(self.channel_sizes[8], self.channel_sizes[9], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[9]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[9]),



            nn.Conv1d(self.channel_sizes[9], self.channel_sizes[10], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[10]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[10]),




            nn.Conv1d(self.channel_sizes[10], self.channel_sizes[11], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[11]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[11]),




            nn.Conv1d(self.channel_sizes[11], self.channel_sizes[12], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[12]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[12]),




            
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[12], self.channel_sizes[13], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[13]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[13]),
            
            nn.Conv1d(self.channel_sizes[13], self.channel_sizes[14], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[14]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[14]),


            nn.Conv1d(self.channel_sizes[14], self.channel_sizes[15], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[15]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[15]),


            )
    
    
    def forward(self, inputs): 
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        #print(inputs.shape)
        output = self.features(inputs) 
        return output

class ConvNetSEClassifier(nn.Module):
    """modification using se blocks for feature extraction instead"""
    def __init__(self,num_classes=26):
        super(ConvNetSEClassifier, self).__init__()
        #9 classsifers
        num_features = 512
        num_mid = 128
        self.features =  ConvNetSEComponent()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = nn.Sequential(nn.Linear(num_features,
                                                  num_mid),
                                        nn.LeakyReLU(),
                                        nn.Linear(num_mid,
                                                  num_mid),
                                        nn.LeakyReLU(),
                                        
                                        nn.Linear(num_mid,
                                                  num_classes))
    def forward(self, inputs,mask=None): 
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
        output = self.global_pool(output)
        
        #print(output.view(act_batch_size,-1).shape)
        output = self.classifier(output.view(act_batch_size, -1))
        return output
    
    def get_features(self,inputs):
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
        output = self.global_pool(output)
        
        return output
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)



################# con1DNet###############
class ConvTransformNet(nn.Module):
    def __init__(self, num_classes=26):
        super(ConvTransformNet,self).__init__()
        #classifier layers
        num_features=128
        #num_features=128
        #num_features=64
        num_mid = 128
        self.features = ConvNetComponent()
        self.global_pool = nn.AdaptiveMaxPool1d(1)


        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=num_features,nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder,num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(num_features, num_mid),nn.LeakyReLU(),nn.Linear(num_mid,num_classes))

    def forward(self, inputs,src_mask=None): 
        output = output.permute(2,0, 1)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output)
        output = output.permute(1,2, 0)

        output=output.reshape(output.shape[0],output.shape[2]*output.shape[1])


        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
        output = self.global_pool(output)
        output = self.transformer(output.view(act_batch_size,1,-1))
        
        output = self.classifier(output.view(act_batch_size, -1))
        return output
    
    def get_features(self,inputs):
        act_batch_size = inputs.size(0)
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)

        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)
class NoraTransform(nn.Module):
    def __init__(self, num_classes=26):
        super(NoraTransform,self).__init__()
        #classifier layers
        num_features=128
        #num_features=128
        #num_features=64
        num_mid = 128
        dropout=0.5
        nhead=8
        nhid=6
        
        self.features = ConvNetComponent()
        self.global_pool = nn.AdaptiveMaxPool1d(1)


        self.model_type = 'NoraTransformer'
        self.pos_encoder = PositionalEncoding(num_features, dropout)
        encoder_layers = TransformerEncoderLayer(num_features, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nhid)

        self.linear=nn.Sequential(nn.Linear(num_features*132, num_features),nn.LeakyReLU())
        self.linear2=nn.Sequential(nn.Linear(num_features, 64),nn.LeakyReLU())
        self.linear3=nn.Sequential(nn.Linear(64, 26),nn.LeakyReLU())
        self.dropout = nn.Dropout(0.3) 
     
    def forward(self, inputs,src_mask=None): 
        act_batch_size = inputs.size(0)
     
        #num features=128
        inputs = inputs.view(act_batch_size, 1, -1)
        output = self.features(inputs)
       
        
        #output = self.global_pool(output)
       
        output = output.permute(2,0, 1)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output)
        output = output.permute(1,2, 0)

        output=output.reshape(output.shape[0],output.shape[2]*output.shape[1])
     
        output=self.linear(output)
        output=self.dropout(output)
        output=self.linear2(output)
        output=self.linear3(output)
        return output


    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

class transformer_resnet34(nn.Module):

    def __init__(self, ntoken, emsize, nhead, nhid, nlayers, dropout=0.2):
        super(transformer_resnet34, self).__init__()
        self.model_type = 'Transformer_resnet34'
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
       # self.encoder = nn.Embedding(ntoken, emsize)
        self.emsize = emsize
        self.previous = models_2020.resnet34(input_channels=1)  
 
        self.linear=nn.Sequential(nn.Linear(512*79, 512),nn.LeakyReLU())
        self.linear2=nn.Sequential(nn.Linear(512, 64),nn.LeakyReLU())
        self.linear3=nn.Sequential(nn.Linear(64, 26),nn.LeakyReLU())
        self.dropout = nn.Dropout(0.3) 

    
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

    def forward(self, src,src_mask=None):
        
        """ 
        expected input shape [(Batch_size, N_channels==1,sequence_length==2570)]
        """
        ####pass the input through resnet34########
      
        src=torch.tensor(src, dtype=torch.float)
        src=self.previous(src)
        
        src = src.permute(2,0, 1)


        src = self.pos_encoder(src)
     
   

        output = self.transformer_encoder(src)
       
        output = output.permute(1,2, 0)
      
  

        output=output.reshape(output.shape[0],output.shape[2]*output.shape[1])
        
        output=self.linear(output)
  
        output=self.linear2(output)
        output=self.dropout(output)
        output=self.linear3(output)

        return output
class PositionalEncoding(nn.Module):
    """
    create a positional encoding on the inputs , 
    expected input shape [(Batch_size, 512)]
    512 represents the Number of features extracted by the previous convolutional layers.

    Returns :
    Postionwise encoded representation of the input
    """
    def __init__(self, d_model, dropout=0.2, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

from torch.optim import Adam


class NoamOptimizer(Adam):

    def __init__(self, params, d_model, factor=2, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9):
        # self.optimizer = Adam(params, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor
        super(NoamOptimizer, self).__init__(params, betas=betas, eps=eps)
    def step(self, closure=None):
        self.step_num += 1
        self.lr = self.lrate()
        for group in self.param_groups:
            group['lr'] = self.lr
        super(NoamOptimizer, self).step()

    def lrate(self):
        return self.factor * self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))*100


from torch.utils.data import Dataset, DataLoader
class records_dataset(Dataset):
    def __init__(self,  files, labels=None):  
        self.records = files
        self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if self.labels is not None:
            return record, self.labels[idx]
        else:
            return record




class transformer_Conv1D(nn.Module):

    def __init__(self, ntoken, emsize, nhead, nhid, nlayers,nchanels, dropout=0.2):
        super(transformer_Conv1D, self).__init__()
        self.model_type = 'transformer_Conv1D'
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
       # self.encoder = nn.Embedding(ntoken, emsize)
        self.emsize = emsize
        
        model = models_2020.Conv1DNet(num_classes = 26,num_channels = 1) 
        self.CV1D =model.conv_model_max
 

        self.linear=nn.Linear(self.emsize*256,512 )
        self.linear2=nn.Linear(512,64 )
        self.linear3=nn.Linear(64,26 )
        self.dropout = nn.Dropout(0.3)

        model_conv1D = models_2020.Conv1DNet(num_classes = 26,num_channels = 1) 
    
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

    def forward(self, src, src_mask):
        """ 
        expected input shape [(Batch_size, N_channels==1,sequence_length==2570)]
        """
        src=torch.tensor(src, dtype=torch.float)
        src=self.CV1D(src)
       
    

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #### Output shape: (Batch,N_features=512,Model_dimention=128)
       
        ############   reshape the output to pass it through a linear layers    #################
        output=output.view(output.shape[0],-1)
        output=self.linear(output)
        output=self.linear2(output)
        output=self.dropout(output)
        output=self.linear3(output)

        return output








class SignLoss(nn.Module):
    '''try out
    sign loss
    arxiv 2101.03895.pdf'''
    def __init__(self,class_weights,weight=None):
        super(SignLoss, self).__init__()
        
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=class_weights,weight=weight)
        self.name=("SignLoss")
    def sg(self,p,y):
        u = abs(p-y)
        return torch.where(u<0.5,y+2*p*y+y**2,torch.ones_like(p))
    
    def forward(self, p,y):
        #p = F.sigmoid(p) 
        l = self.sg(p,y)*self.BCELoss(p,y)
        l = l.mean()
        return l
class DiceSignLoss(nn.Module):
    def __init__(self, classes_weights=None,weight=None):
        super(DiceSignLoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=classes_weights,weight=weight)
        self.name=("DiceSignLoss")
    def sg(self,p,y):
        u = abs(p-y)
        return torch.where(u<0.5,y+2*p*y+y**2,torch.ones_like(p))
    

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs) 
        l = self.sg(inputs,targets)*self.BCELoss(inputs,targets)
        l = l.mean()
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        Dice_sign = 0.5*l + 0.5*dice_loss
        return l

 
class DiceBCELoss(nn.Module):
    def __init__(self, classes_weights=None,weight=None):
        super(DiceBCELoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=classes_weights,weight=weight)
        self.name=("DiceBCELoss")
    def forward(self, inputs, targets, smooth=1):
        
        BCE = self.BCELoss(inputs, targets)
        inputs = F.sigmoid(inputs)
        
        

        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        Dice_BCE = BCE + dice_loss
    
        
        return Dice_BCE



class DiceLoss(nn.Module):
    def __init__(self, classes_weights=None,weight=None,size_average=True):
        super(DiceLoss, self).__init__()
        self.name=("DiceLoss")
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        print(dice) 
        
        return 1 - dice






