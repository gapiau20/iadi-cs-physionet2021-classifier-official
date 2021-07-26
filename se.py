
class ConvNetSEComponent(nn.Module):
    """
    feature extraction from Nora's ConvNet
    (see CNN, https://arxiv.org/abs/1912.00852)
    modified by adding squeeze excitation layers
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self):
        super(ConvNetSEComponent, self).__init__()

        self.channel_sizes = [16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128]
        self.channel_sizes = [16, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
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
            nn.Dropout(p=0.5),
            
            nn.Conv1d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[2]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(self.channel_sizes[2], self.channel_sizes[3], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[3]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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
            
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(self.channel_sizes[8], self.channel_sizes[9], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[9]),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            SEInceptionLayer(self.channel_sizes[9]),
            
            nn.Conv1d(self.channel_sizes[9], self.channel_sizes[10], kernel_size=kernel_size), #,stride=(2,1)
            nn.BatchNorm1d(self.channel_sizes[10]),
            nn.ReLU(),
            SEInceptionLayer(self.channel_sizes[10]),

            #nn.Dropout(p=0.5)
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
        num_features = 256
        num_mid = 128
        self.features =  ConvNetSEComponent()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        #self.classifier = nn.Linear(num_features, num_classes)
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
