""" dataloader"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wfdb import processing
from scipy import signal
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from helper_code import *
from records_preparation import *
from stratified_kfold import *

def select_database(header_list, record_list, one_hot_labels, database=None):
    """
    filter the fold to get only the elements that belong to the specified database
    parameters : 
    headerlist, recordlist : lists of .hea and .mat files in the fold
    one_hot_labels : tensor of the one hot labels of the files (1 file per line)
    database : if None, the function returns the complete fold
    otherwise, string corresponding to the tag of the database : 
    "
    "E" georgia
    "HR" ptb xl
    "S" 
    "JS" chapman (the new one)
    "Q" cpsc complement
    "A" cpsc
    """
    if database==None:
        #return the full dataset of the fold
        return header_list, record_list, one_hot_labels
    else :
        headers, records, labels = [],[],[]
        for idx in range(len(header_list)):
            if database in header_list[idx]:
                headers.append(header_list[idx])
                records.append(record_list[idx])
                labels.append(one_hot_labels[idx,:])
        labels = np.array(labels)
        return headers, records, labels
                

class OneLeadDataset(Dataset):
    def __init__(self, header_list, record_list, one_hot_labels, data_folder, transform=None,device='cpu'):
        self.transform=None
        self.records = []
        self.ecg = []
        self.labels = []
        self.duration=10#seconds
        self.target_fs =250#Hz (incart is 257Hz)
        self.transform=transform
        self.order=3
        self.bandpass=[0.5,45]#Hz

        #for each header
        print("total headers to load")
        print(len(header_list))
        
        for i in range(len(header_list)):
        #for i in range(2):    #test
            header_name =header_list[i]
            record_name = record_list[i]
            record_label = one_hot_labels[i,:]

            #load header
            header = load_header(header_name)
            recording = load_recording(record_name)
            #get leads
            leads = get_leads(header)
            num_leads = len(leads)
            
            #labels and record names
            for lead_i in range(num_leads) : 
                self.records.append(record_name)
                self.labels.append(torch.tensor(record_label))
                
            #prepare record tensor
            self.ecg.append(prepare_input_record(header,recording, leads))
            #standardization is performed during the prepare_input_record
            
        self.ecg = torch.stack(self.ecg)
        self.ecg = self.ecg.flatten(start_dim=0,end_dim=1).double()
        self.labels=torch.stack(self.labels)
        self.target_pred = self.labels.clone().detach() #for custom loss update
        
        return
    
    def __getitem__(self, idx):
        if self.transform : 
            sample = self.transform(self.ecg[idx])
        else : 
            sample = self.ecg[idx]
        return sample, self.labels[idx], self.target_pred[idx]
    
    def __len__(self):  
        return len(self.labels)

    def update_target_pred(self,idx, pred, momentum=0.9):
        """
        update target prediction for 
        https://arxiv.org/pdf/2105.13244.pdf
        momentum : float btween 0 and 1
        """
        self.target_pred[idx] = momentum*target_pred[idx]+(1-momentum)*pred
        return
    
if __name__=='__main__' : 
    train_data_dir = "training_data/"+"full_set"
    test_data_dir = "test_data/"
    model_dir = "model/"
    test_outputs_dir = "test_outputs/"

    #get challenge files
    os.chdir('/physionet/')
    challenge_files = find_challenge_files(train_data_dir)
    challenge_files_array = np.array(challenge_files)

    #remove incart
    header_files_filtered = [str(i) for i in challenge_files_array[0] if not ("I" in i)]
    record_files_filtered = [str(i) for i in challenge_files_array[1] if not ("I" in i)]
    
    
    num_recordings=len(record_files_filtered)
    #extract classes and labels
    classes,num_classes = extract_classes(header_files_filtered)
    # reduce the data to the scored classes
    mapping_csv_file = '/physionet/dx_mapping_scored.csv'
    scored_classes = filter_scored_classes(classes, mapping_csv_file)
    
    num_classes_filtered = len(scored_classes)
    labels_unfiltered = extract_labels(header_files_filtered, classes, num_recordings)
    labels_filtered = extract_labels(header_files_filtered, scored_classes,
                                     num_recordings)
    

    # #create kfold
    # print('make folds')
    # fold_list = stratified_k_fold(labels_filtered,1)
    

    
    # train_fold_labels=[labels_filtered[i,:] for i in train_fold_list]
    # train_fold_headers= split_list(header_files_filtered, train_fold_list)
    # train_fold_records= split_list(record_files_filtered, train_fold_list)
    
    # test_fold_labels = [labels_filtered[i,:] for i in test_fold_list]
    # test_fold_headers = split_list(header_files_filtered, test_fold_list)
    # test_fold_records = split_list(record_files_filtered, test_fold_list)

    print('make dataset')
    # databases=[None]

    dataset=OneLeadDataset(header_files_filtered, record_files_filtered,labels_filtered,train_data_dir)

    print('mean')
    print(dataset.mean())
    print('std')
    print(dataset.std())
