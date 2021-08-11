import torch.nn as nn
import torch
import new_model
import utils

architectures = nn.ModuleDict({'se':new_model.ConvNetSEClassifier()})
model_names = ['se','se']
model_param_file = 'best_se_dicebce_val_set_calib_expt.pt'
model_dir = './model'
model_params = [model_param_file,model_param_file]

ensemble = new_model.EnsembleModel(architectures, model_names,model_params,model_dir)


x = torch.ones((1,12,2500))
print(x.size())
print('apply ensemble model')
y = ensemble(x)
print(y.size())
print('tryout multilead predict on ensemble model')
z = utils.multilead_predict(x,ensemble,0.5)
print('z',z)
