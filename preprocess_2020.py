import argparse
from pathlib import Path

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import scipy.signal as sig
from scipy import stats
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import torch
from numba import jit, cuda
import pickle
from wfdb import processing #qrs detection, resampling, etc
from sklearn.decomposition import PCA
from tqdm import tqdm

from helper_code import *
import utils
# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes_challenge(equivalent_classes,challenge_classes):
    classes = set()
    for arr in challenge_classes:
        for i in range(len(equivalent_classes)):
            if(arr==equivalent_classes[i][0] or arr==equivalent_classes[i][1]):
                arr=equivalent_classes[i][1]
        classes.add(arr)
    return sorted(classes)
def get_classes(input_directory, filenames,challenge_classes):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        if c in challenge_classes:

                            classes.add(c.strip())
    return sorted(classes)
    
#find sampling frequency and nb of samples.
def get_infos(filename):
    infos = dict()
    with open(filename, 'r') as f:
        for l in f:
            fe=l.split(' ')[2]
            N=l.split(' ')[3]
            break
    return int(fe), int(N)

def get_infos_test(header):
    infos = dict()
    fe=header.split(' ')[2]
    N=header.split(' ')[3]
    leads = get_leads(header)
    return int(fe), int(N),leads

def center_pad(sig,duration,fs):

    if sig.shape[-1]>duration*fs:
        start = sig.shape[-1]//2-duration*fs//2
        stop = start+duration*fs
        x = sig[start:stop]
    else : 
        x = torch.zeros(duration*fs)
        start = x.shape[-1]//2-sig.shape[-1]//2
        stop = start+sig.shape[-1]
        x[start:stop]=sig
    return x

#Preprocess_ECG_signals
def preprocess(recording, fe, target_fs, duration, adc, baseline, leads):
    """"""
    indices = list()
    available_leads=leads
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording=recording[indices,:]
    num_leads = len(leads)
    
    order=3
    bandpass = [0.5,120]
    record_tensor = []
    for lead_i in range(num_leads):
        #correct the voltage and baseline 
        ecg = (recording[lead_i,:]-baseline[lead_i])/adc[lead_i]

        #butter
        nyq = fe/2
        band=np.array(bandpass)/nyq
        b,a = signal.butter(order,band,'bandpass')
        #ecg_filtered=signal.filtfilt(b,a,ecg)
        ecg_filtered = signal.lfilter(b,a,ecg)

        #notch 60Hz
        fn60=60#hz
        bw=5#hz
        Q=fn60/bw
        b_notch60,a_notch60=signal.iirnotch(fn60,Q,fs=fe)
        #ecg_notch60=signal.filtfilt(b_notch60,a_notch60,ecg_filtered)
        ecg_notch60 = signal.lfilter(b_notch60,a_notch60,ecg_filtered)

        #notch 50Hz
        fn50 = 50
        bw=5
        Q=fn50/bw
        b_notch50,a_notch50=signal.iirnotch(fn50,Q,fs=fe)
        #ecg_notch50 = signal.filtfilt(b_notch50,a_notch50,ecg_filtered)
        ecg_notch50 = signal.lfilter(b_notch50,a_notch50,ecg_filtered)

        #resample 250Hz
        ecg_resampled = processing.resample_sig(ecg_notch50,fe,target_fs)[0]
        
        #standardize
        ecg_unscaled = torch.tensor(ecg_resampled)
        m,C =ecg_unscaled.mean(), max(ecg_unscaled.std(),1e-5)
        #m,C = ecg_unscaled.min(), max(ecg_unscaled.max()-ecg_unscaled.min(),1e-5)
        #m,C = ecg_unscaled.median(),max(ecg_unscaled.max()-ecg_unscaled.min(),1e-5)
        ecg_scaled = (ecg_unscaled-m)/C

        #center pad
        ecg_scaled = center_pad(ecg_scaled,duration,target_fs)
        record_tensor.append(ecg_scaled)






    record_tensor = torch.stack(record_tensor)


    return np.array(record_tensor)
    


def ecg_preprocessing(input_directory, classes_weights_file):
    #get headers
    challenge_files = find_challenge_files(input_directory)
    challenge_files_array  = np.array(challenge_files)
    header_files = [str(i) for i in challenge_files_array[0]]

    print('make classes and target vector')
    weights_file = classes_weights_file
    classes,weights = utils.load_weights(weights_file)
    saved_classes = classes
    labels= torch.tensor(utils.load_labels(header_files,classes)).float()
    print('labels')
    print(labels.size())


    num_classes = len(classes)
    num_files = len(header_files)
    recordings = []
    label_list = []
    
    # #preprocess the record
    duration=10#seconds
    target_fs = 250#Hz (incart is 257Hz)

                                                                                
    labels_array = np.zeros((num_files,num_classes))
    print('labels_array  shape',labels_array.shape)
    #for each record
    for i in range(num_files):
        
        #load header and get header information
        recording, header = load_challenge_data(header_files[i])
        head = load_header(header_files[i])
        available_leads = get_leads(head)
        num_leads=len(available_leads)
        leads=available_leads
        fe,N= get_infos(header_files[i])
        adc, baseline = get_adc_gains(head, leads), get_baselines(head, leads)
        
        #preprocess recording
        preprocessed_recording=preprocess(recording, fe, target_fs, duration,adc, baseline,leads)
        recordings.append(preprocessed_recording)
        label_list.append(labels[i,:])
#        for ld in range(num_leads) :


    #stack recordings and put into tensor
    recordings_array = np.array(recordings)
    labels_tensor = torch.stack(label_list)
    print(labels_tensor.size())
    print('dataset distrib')
    print(classes)
    print(labels_tensor.sum(dim=0))
    
    recordings_tensor = torch.from_numpy(recordings_array).float()
 
    return(recordings_tensor,labels_tensor,classes, weights)




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Script for preprocessing the data")
    parser.add_argument("--input",
                        type=Path,
                        help="Path to the tensors data",
                        default="/home/mouin/Physionet/python-classifier-2021-main/data")
    parser.add_argument("--output",
                        type=Path,
                        help="Path to the tensors data",
                        default="/home/mouin/Physionet/python-classifier-2021-main/data")

    args = parser.parse_args()

    input_directory= args.input
    tensors_directory= args.output

    #le preprocessing général
    recordings_array,recordings_tensor,labels_array,labels_tensor,classes = ecg_preprocessing(input_directory, tensors_directory)
    
 
