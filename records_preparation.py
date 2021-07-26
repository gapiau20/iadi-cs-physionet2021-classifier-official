import os
import itertools


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from helper_code import *
from stratified_kfold import *


from wfdb import processing #qrs detection, resampling, etc
from scipy import signal
from sklearn import preprocessing

#equivalent classes mapping for scored classes grouping
grouping_dict = {
    '59118001':'713427006',
    '63593006':'284470004',
    '17338001':'427172004',
}

def extract_classes(header_files):
    """extract all the available classes (snomed ct code) from 
the specified header files
parameters : header_files list of headers 
"""
    print("extract_classes(header_files)")
    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        header_labels =get_labels(header) 
        classes |= set(header_labels)
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    return classes, num_classes

def filter_scored_classes(classes, mapping_csv, class_dict=grouping_dict):
    """
    from all the classes, extract only the classes that are scored
    classes : the classes set to filter
    mapping_csv : str name of the csvfile containing the classes (from the challenge evaluation code)
    class_dict : dictionary mapping the equivalent classes (cf grouping_dict above)
    """
    classes_df = pd.read_csv(mapping_csv)
    tested_classes = classes_df['SNOMED CT Code']
    filtered_classes = [str(i) for i in tested_classes if str(i) in classes]
    return filtered_classes

def class_weight(csv_file,order=1):
    '''
    compute class weights for the trainin
    parameters : 
    csv file (dx_mapping scored or unscored.csv)
    order (default 1) : exponent
    
    '''
    class_df = pd.read_csv(csv_file)
    pos = torch.tensor(class_df['Total'])
    tot = pos.sum()
    
    Wn = ((tot-pos)/pos)**order

    return Wn

def extract_labels(header_files, classes, num_recordings):
    """
    one hot encoding of the classes 
    the grouping of the equivalent classes is done at this step using the grouping dict
    header_files : the list of headers
    classes : the set of scored snomed ct codes
    num_recordings : number of recordings
    """
    num_classes = len(classes)
    labels = np.zeros((num_recordings, num_classes)) # One-hot encoding of classes
    for i in range(num_recordings):
        header = load_header(header_files[i])
        current_labels = get_labels(header)
        for label in current_labels:
            #if key is in grouping dict (ie in the classes to merge)
            if label in grouping_dict.keys():
                label =  grouping_dict[label]#relabel according to grouping dict
            #encode the one hot of the class
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1
    return labels


def split_list(list_to_split,fold_list):
    fold_list = [[list_to_split[idx] for idx in fold_idx] for fold_idx in fold_list]
    return fold_list

def complementary_list(folds,idx):
    """from the fold at position idx in folds, extract the complementary fold"""
    complementary = folds.copy()
    complementary.pop(idx)
    complementary = list(itertools.chain.from_iterable(complementary))
    return complementary


def center_pad(sig,duration,fs):

    if sig.shape[-1]>duration*fs:
        start = sig.shape[-1]//2-duration*fs//2
        stop = start+duration*fs
        x = sig[start:stop]
    else : 
        x = np.zeros(duration*fs)
        start = x.shape[-1]//2-sig.shape[-1]//2
        stop = start+sig.shape[-1]
        x[start:stop]=sig
    return x

    
def prepare_input_record(header,recording, leads):    


    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording=recording[indices,:]

    num_leads = len(leads)
    #get meta information for the leads
    fs, duration = get_frequency(header),get_num_samples(header)
    adc, baseline = get_adcgains(header, leads), get_baselines(header, leads)

    # #preprocess the record
    # duration=10#seconds
    # target_fs = 250#Hz (incart is 257Hz)
    duration = 10
    target_fs = 250
    
    order=3
    bandpass = [0.5,45]
                                                                                

    record_tensor = []
    num_leads = len(leads)

    
    for lead_i in range(num_leads):
        #correct the voltage and baseline 
        ecg = (recording[lead_i,:]-baseline[lead_i])/adc[lead_i]
        #resample to 250Hz
        ecg = processing.resample_sig(ecg,fs,target_fs)[0]

        #butterworth
        nyq = target_fs/2
        band = np.array(bandpass)/nyq
        b,a = signal.butter(order, band, 'bandpass')
        ecg_filtered = signal.filtfilt(b,a,ecg)

        #extract segment of the record
        ecg_filtered = center_pad(ecg_filtered,duration,target_fs)
        #record_tensor.append(torch.tensor(ecg_filtered.copy()))
        
        #standardize
        eps=1e-4 #case std=0
        C = max(ecg_filtered.std(),eps)
        ecg_stdardized = ecg_filtered-ecg_filtered.mean()/C
        record_tensor.append(torch.tensor(ecg_stdardized.copy()))


    record_tensor = torch.stack(record_tensor)
    return record_tensor



if __name__=="__main__":
    
    train_data_dir = "training_data/"+"full_set"
    test_data_dir = "test_data/"
    model_dir = "model/"
    test_outputs_dir = "test_outputs/"

    #get challenge files
    os.chdir('/physionet/')
    challenge_files = find_challenge_files(train_data_dir)
    challenge_files_array = np.array(challenge_files)
    #shuffle data
    challenge_files_array
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
    
    print('extracted classes from header')
    print(labels_filtered.shape)
    print('scored classes')
    print('test fold stratification')
    #create kfold 
    fold_list = stratified_k_fold(labels_filtered,10)
    #check the distribution of the kfolds
    print(len(fold_list), "folds")

    label_distrib = np.sum(labels_filtered,axis=0)/np.sum(labels_filtered)
    #plt.matshow(label_distrib.reshape(1,len(label_distrib)))
    
    folds = [labels_filtered[i,:] for i in fold_list]
    fold_distrib = [np.sum(i,axis=0) for i in folds]
#    fold_distrib = [np.sum(i,axis=0)/np.sum(i) for i in folds]

    for i in fold_distrib : 
       # plt.matshow(i.reshape((1,len(i))))
        plt.figure()
        sns.heatmap(i.reshape((1,len(i))).astype(int),annot=True,fmt='d')
#    plt.matshow((fold_distrib[1]-fold_distrib[0]).reshape(1,num_classes_filtered))
    for i in fold_distrib:
        print(np.max(np.abs(i-label_distrib)))

    fold_headers = [[header_files_filtered[idx] for idx in fold_idx]for fold_idx in fold_list]
    fold_records = [[record_files_filtered[idx] for idx in fold_idx]for fold_idx in fold_list]
    print('test split_list')
    fold_headers_ = split_list(header_files_filtered, fold_list)
    print('comparing len of folds')
    print([len(i) for i in fold_headers])
    print([len(i) for i in fold_headers_])
