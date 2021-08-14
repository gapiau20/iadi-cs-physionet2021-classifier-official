#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################
from sklearn import impute
from torch.nn.modules.activation import Threshold
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from preprocess_2020 import *
import new_model
from new_model import records_dataset
import exp_supervision  
import records_preparation
import stratified_kfold
# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
##########################
##   Import Libraries   ##
##########################
import warnings
warnings.filterwarnings('ignore')
import numpy as np, os, sys, joblib
import pickle
import json
import random
import argparse
from pathlib import Path
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from  torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts,CyclicLR,LambdaLR
from models_2020 import Conv1DNet, Conv2DNet,resnet18, resnet34, resnet50, resnet101
import utils
from datetime import datetime
from new_model import  NoamOptimizer
#Setting the random generator to default
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    expt_dir = '/physionet/expt_logs/'
    expt_files = sorted(os.listdir(expt_dir))
    print(expt_files)
    architectures, model_names, thresholds = [],[],[]
    for f in expt_files : 
        expt = exp_supervision.ExpSupervisor(data_dir=data_directory,model_dir=model_directory)
        expt.load(os.path.join(expt_dir,f))
        arch, mod_name, thr = run_experiment(data_directory, model_directory,expt)
        architectures.append(arch)
        model_names.append(model_names)
        thresholds.append(thresholds)
    


    # n_cpu=4
    # pool = mp.Pool(n_cpu)
    # pool.starmap(launch_expt,[(data_directory,model_directory,expt_dir,f) for f in expt_files])
    
    
    return
def launch_expt(data_directory,model_directory,expt_dir,f):
    '''routine for the multithreading, calls run_experiment
    bug
     f : experiment filename (.json)
    '''
    expt = exp_supervision.ExpSupervisor(data_dir=data_directory,model_dir=model_directory)
    expt.load(os.path.join(expt_dir,f))
    run_experiment(data_directory, model_directory,expt)
    return
#################################################################################################
# run experiment
#################################################################################################
def run_experiment(data_directory,model_directory,expt):
    #create the experiment object containing all the parameters


    # Find header and recording files.
    #Time of execution calculation
    start_time = datetime.now()

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)
    # Train a model for each lead set.
    
    # Tensors directories Directories
    #tensors_directory = "/home/mouin/Physionet/python-classifier-2021-main/data"

    # Hyperparameters
    device = torch.device('cuda')
    model_name =expt.expt_tag
    num_epochs = expt.num_epochs
    batch_size = expt.batch_size


    kept_classes = expt.kept_classes
    shuffle_dataset = expt.shuffle_dataset
    validation_split = expt.validation_split
    

    leads = expt.leads
    list_leads = expt.list_leads
    
    ##################
    ## Loading Data ##
    ##################
    print('Loading data...')
   

    equivalent_classes = expt.equivalent_classes
    
    #preprocess the directory
    normal_class = expt.normal_class
    scoring_weights_csv = expt.scoring_weights_csv
    dx_mapping_csv = expt.dx_mapping_csv
    recordings_tensor,labels_tensor,classes, weights= ecg_preprocessing(data_directory,
                                                                                         scoring_weights_csv)
    
    #prepare class weights
    challenge_classes = [next(iter(i)) for i in classes]
    saved_classes = classes
    pos_weight = utils.class_weight(dx_mapping_csv,challenge_classes)
    #rescale the recordings
    recordings_rescaling = [recordings_tensor.mean(), recordings_tensor.std()]#min,max
    recordings_tensor = (recordings_tensor-recordings_rescaling[0])/recordings_rescaling[1]

    print("Recordings tensor shape : ",recordings_tensor.shape)
    
    #get headers files
    challenge_files = find_challenge_files(data_directory)
    challenge_files_array  = np.array(challenge_files)
    header_files = [str(i) for i in challenge_files_array[0]]

    
    print('Creating DataLoaders...')
    
    #########################################
    ## Train, valid Data loader definition ##
    #########################################
    tmp=recordings_tensor
    #80% train / 20% Val_Test With Equal Distributions (5 folds)
    fold_list = stratified_kfold.stratified_k_fold(labels_tensor.numpy(),expt.n_folds,shuffle=False)
    fold_idx = expt.fold_idx#1 fold iteration
    train_idx = records_preparation.complementary_list(fold_list,fold_idx)
    val_idx = fold_list[fold_idx]


    
    print('first fold iteration')
    print('training size ',len(train_idx))
    print('val size',len(val_idx))
    #get the indices where the files are
    train_headers = [header_files[i] for i in train_idx]
    tmp_train=[tmp[i] for i in train_idx]
    labels_train=[labels_tensor[i] for i in train_idx]


    databases = expt.databases
    #lists
    train_headers,tmp_train,labels_train,batch_weights=utils.order_databases(
        train_headers,tmp_train,labels_train,
        databases)

    if expt.weight_batches==True:
        print('weighting_batches')
        batch_weights=utils.weight_db_classes(train_headers,batch_weights,dx_mapping_csv,challenge_classes)
        #tryout :

    else :
        print('not weighting batches')
    print('batch_weights shape',batch_weights[0].size())
    labels_val=[labels_tensor[i] for i in val_idx]
    tmp_val=[tmp[i] for i in val_idx]
    

  
    ####################reshape the train tensors (records, labels, weights)##################
    #stack the lists into tensors
    tmp_train=torch.stack(tmp_train)
    labels_train=torch.stack(labels_train)
    batch_weights=torch.stack(batch_weights)

    #tryout normalization labels
    tot_active_labels = labels_train.sum(dim=1).view(-1,1).repeat(1,labels_train.size(1))
    #tot_inactive_labels = tot_active_labels.size(1)-tot_active_labels
    #batch_weights*=((1+tot_active_labels)/(1+tot_inactive_labels))
    #batch_weights*=((1+tot_inactive_labels)/(1+tot_active_labels))
    batch_weights*=1/torch.maximum(torch.ones_like(tot_active_labels),tot_active_labels)
    
    records_train=tmp_train[:,0,:]
    new_labels_train=labels_train
    new_weights_train=batch_weights
    
    for i in range(1,tmp_train.shape[1]):
        records_train=torch.cat((records_train,tmp_train[:,i,:]),0)
        new_labels_train=torch.cat((new_labels_train,labels_train))
        new_weights_train=torch.cat((new_weights_train,batch_weights))
    print('size new_labels_train',new_labels_train.size())
    print('size new_weights_train',new_weights_train.size())
    print('new_labels_train sum', new_labels_train.sum(dim=0))
    print('records_train shape',records_train.size())
    print('each lead is considered as 1 sample (batch dimension)')
    records_train=torch.reshape(records_train, (records_train.shape[0], 1,records_train.shape[1]))
 

    dataset_train = records_dataset(
        files=records_train,
        labels=new_labels_train,
        weights=new_weights_train
                             )

    ####################reshape the validation tensors##################
    tmp_val=torch.stack(tmp_val)
    labels_val=torch.stack(labels_val)
    records_val=tmp_val
    new_labels_val=labels_val
    print('new_labels_val sum',new_labels_val.sum(dim=0))
    print('each record is considered as a sample')
    dataset_val = records_dataset(
        files=records_val,
        labels=new_labels_val
                              )


##########################
## DataLoaders Creation ##
##########################

    #labels_tensor_val_test = torch.index_select(labels_tensor, 0, torch.tensor(val_test_idxs)) #Here we create a new labels tensor for test_val sets
    print(f'# of training samples = {new_labels_train.shape[0]}, # of Valid samples = {new_labels_val.shape[0]}')

    trainloader = torch.utils.data.DataLoader(dataset_train,  batch_size=batch_size,shuffle=True)
    validloader = torch.utils.data.DataLoader(dataset_val,batch_size=batch_size,shuffle=False)
    ##########################
    ## Model Initialization ##
    ##########################
    ntoken=expt.ntoken
    emsize=expt.emsize
    nhead=expt.nhead
    nhid=expt.nhid
    nlayers=expt.nlayers
    nchanels=expt.nchanels
    dropout=expt.dropout

    models = nn.ModuleDict({
#        'nora':new_model.OldNora_ConvNet(),
 #       'noraTransform':new_model.NoraTransform(),
        'se':new_model.ConvNetSEClassifier(dropout=expt.dropout),
  #      'seTransform':new_model.SETransformer()
    })
    model = models[expt.model]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # if expt.retraining:
    #     print('starting from a pretrained model : ',expt.pretrained_model_name)
    #     utils.load_pretrained_model(model, expt.pretrained_model_name,model_directory)

    # Loss and optimizer
    pos_weight = pos_weight.to(device)#class weight to same device as the other tensors
    print(pos_weight.size())


    criteria = {
#        'bcelogits':nn.BCEWithLogitsLoss(pos_weight=pos_weight),
#        'sign':new_model.SignLoss(class_weights=pos_weight),
        'dicebce':new_model.DiceBCELoss(classes_weights=pos_weight),
#        'dicesign':new_model.DiceSignLoss(classes_weights=pos_weight)
    }
    
    criterion = criteria[expt.criterion]

    optimizers= {
        'sgd':torch.optim.SGD(model.parameters(), lr=expt.adam_lr, momentum=0.9),
#        'adam':torch.optim.Adam(model.parameters(), lr=expt.adam_lr),
#        'adam_l2':torch.optim.Adam(model.parameters(),lr=expt.adam_lr, weight_decay=expt.adam_weight_decay),
#        'noam':NoamOptimizer(model.parameters(), d_model=expt.noam_d_model)
    }
    optimizer = optimizers[expt.optimizer] 

    #schedulers for each epoch

    schedulers = {
        'step':StepLR(optimizer, step_size=expt.step_step_size, gamma=expt.step_gamma),

    }
    scheduler = schedulers[expt.scheduler]
    #schedulers for every batch iteration
    lr_schedulers={
        'none':None,
        'cyclic':CyclicLR(optimizer,base_lr=expt.cyclic_base_lr,max_lr=expt.cyclic_max_lr),

       # 'cos':CosineAnnealingWarmRestarts(optimizer,expt.cos_T0,eta_min=expt.cos_eta_min),
        }
    lr_scheduler = lr_schedulers[expt.lr_scheduler]
    
    ################################
    ## Tensorboard Initialization ##
    ################################

    top_logdir = "./tensorboard_history"
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)

    logdir = utils.generate_unique_logpath(top_logdir, "Inserm_" + model_name +"_leads : " + str(leads)+"_")
    logdir = utils.generate_unique_logpath(top_logdir, expt.expt_tag)
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Where to save the logs of the metrics:
    history_file = open(logdir + '/history', 'w', 1)
    history_file.write("Epoch\t\tTrain loss\t\tTrain acc\t\tTrain f1\t\tTrain C-Metric\t \t Val loss\t\tVal acc\t\tVal f1\t\tVal C-Metric\t\t Learing_rate\n")

    # Generate and dump the summary of the model:
    summary_text = utils.model_summary(model, optimizer, logdir=logdir)

    # Initialise tensorboard SummaryWriter:
    writer = SummaryWriter(log_dir=logdir)
    writer.add_text("Experiment summary", summary_text)


    model_path = model_directory+ "/best_" + model_name + ".pt"
    model_checkpoint = utils.ModelCheckpoint(model_path)
        #############################
    ## Training and Validation ##
    #############################        
    print('Training model...')
    best_metric_val = None
    for epoch in range(num_epochs):
        model.train()
        
        train_loss, train_acc,train_f1,train_cmetric = utils.train(model, trainloader, criterion, optimizer,lr_scheduler ,device, weights, classes, normal_class,model_name,writer)
        print(f'Training : Epoch [{epoch+1}/{num_epochs}],Loss [{train_loss:.4f}], Accuracy [{train_acc:.4f}], F1_score [{train_f1:.4f}], Challenge_metric [{train_cmetric:.4f}]')

        
        model.eval()
        confusion_matrix, val_loss, val_acc,val_f1,val_cmetric = utils.valid(model, validloader, criterion, device, weights, classes, normal_class,model_name)#chalenge_classes
        print(f'Validation : Epoch [{epoch+1}/{num_epochs}], Loss [{val_loss:.4f}], Accuracy [{val_acc:.4f}], F1_score [{val_f1:.4f}], Challenge_metric [{val_cmetric:.4f}]')
        
        #Print_confusion_matrix
        #print('Printing confusion matrix of epoch'+str(epoch))

        #save best model
        if best_metric_val is None or val_cmetric>best_metric_val:
        #if True:
            print('in training_code() save better model')
            torch.save(model.state_dict(),model_path)
            print(model_path)
            best_metric_val=val_cmetric
        else :
            print('not best metric--> don\'t save model')
            print('val_cmetric',val_cmetric,'best_metric_val',best_metric_val)

        
        # fig, ax = plt.subplots(7, 4, figsize=(10, 8))
        # for axes, cfs_matrix, classe_name in zip(ax.flatten(), confusion_matrix, challenge_classes):
        #     utils.print_confusion_matrix(cfs_matrix, axes, classe_name, ['N','Y'])
        # fig.text(0.5, 0.04, 'True', ha='center')
        # fig.text(0.04, 0.5, 'Predicted ', va='center', rotation='vertical')
        # plt.savefig('confusion_matrix_epoch_'+str(epoch)+'.png')
        
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))

        # history_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch,
        #                                                 train_loss, train_acc,train_f1,train_cmetric,
        #                                                 val_loss, val_acc,val_f1,val_cmetric,optimizer.param_groups[0]['lr']))

    
        #########################################################################
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation',val_loss , epoch)
        #writer.add_scalar('Acuuracy/train', train_acc, epoch)
        #writer.add_scalar('Acuuracy/validation', val_acc, epoch)
        #writer.add_scalar('training_F1', train_f1 , epoch) 
        writer.add_scalar('ChallengeMetric/train', train_cmetric , epoch) 
       
        
        #writer.add_scalar('validation_F1', val_f1 , epoch) 
        writer.add_scalar('ChallengeMetric/validation', val_cmetric , epoch) 
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'] , epoch) 
    
    end_time = datetime.now()
    print('Time of execution of train.py is : {}'.format(end_time - start_time))
    #Print_confusion_matrix
    


    lead_set12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_set6 = [ 'I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    lead_set4 = ['I', 'II', 'III', 'V2']
    lead_set3 = ['I', 'II', 'V2']
    lead_set2 = [ 'I', 'II']

    #model calibration
    thresholds_opt = 0.5
    thresholds_opt = utils.calibrate(model,validloader,weights,classes,normal_class,device)


    
    save_model(model_directory, leads, saved_classes,  model, model_name, recordings_rescaling,thresholds_opt)    
    return expt.model, model_name, thresholds_opt
################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. Do *not* change the arguments of this function.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    rescaling = model['rescaling']

    #load classifier
    classifier = model['classifier']
    #print(classifier)
    #load classifier model state dict
    model_path = os.path.join(model['model_dir'],'best_'+model['model_name']+".pt")
    print(model_path)
    classifier = new_model.ConvNetSEClassifier()
    
    classifiers = nn.ModuleDict({
#        'nora':new_model.OldNora_ConvNet(),
#        'noraTransform':new_model.NoraTransform(),
        'se':new_model.ConvNetSEClassifier(),
        #'seTransform':new_model.SETransformer()
    })
    classifier = classifiers['se']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.load_state_dict(torch.load(model_path))
    #Threshold=0.5
    #Threshold = 1e-15
    Threshold=torch.tensor(model['thresholds']).to(device)
    classifier = classifier.to(device)
    
    #turn into evaluation mode, and don't track gradients
    
    classifier.eval()
    with torch.no_grad():
        # #data preprocessing
        fe,N,leads= get_infos_test(header)
        num_leads=len(leads)
        target_fs = 250
        duration = 10

        adc, baseline = get_adc_gains(header, leads), get_baselines(header, leads)
        preprocessed_recording=preprocess(recording, fe,target_fs, duration,adc,baseline,leads)
        recordings_tensor = torch.from_numpy(preprocessed_recording).float()
        recording_tensor = (recordings_tensor-rescaling[0])/rescaling[1]

        sz = recording_tensor.size()    
        tmp=recording_tensor.view(1,sz[0],sz[1])

        #predict the output
        probabilities,final_labels = utils.multilead_predict(tmp,classifier,Threshold,device)
        

        final_labels = final_labels.detach().to('cpu').numpy().flatten()
        probabilities = probabilities.detach().to('cpu').numpy().flatten()

        #for debugging purpose
        #plot the ecg record
        # fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        # ax.plot(recordings_tensor.T)
        # fig.savefig("/physionet/ecg.png")
        


    return classes, final_labels, probabilities

################################################################################
#emac
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(model_directory, leads, classes, classifier,model_name="",rescaling=[], thresholds=0.5):
    
    classes = [next(iter(i)) for i in classes]
    d = {'leads': leads, 'classes': classes,  'classifier': classifier,
         "model_name":model_name,'rescaling':rescaling, 'thresholds':thresholds}#'imputer': imputer,
    filename = os.path.join(model_directory, get_model_filename(leads))
    
    joblib.dump(d, filename, protocol=0)

# Load a trained model. This function is *required*. Do *not* change the arguments of this function.
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    model = joblib.load(filename)
    model['model_dir'] = model_directory
    return model

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    return 'model_' + '-'.join(sort_leads(leads)) + '.sav'

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms
