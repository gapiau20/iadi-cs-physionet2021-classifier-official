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
from sklearn.impute import SimpleImputer

from tqdm import tqdm
from helper_code import *
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
        x = np.zeros(duration*fs)
        start = x.shape[-1]//2-sig.shape[-1]//2
        stop = start+sig.shape[-1]
        x[start:stop]=sig
    return x


def ecg_preprocessing_test(header, recording):
    fe,N= get_infos_test(header)
    # headers.append(header)
    preprocessed_recording=preprocess(recording, fe, N)
    if preprocessed_recording.shape[1] > 2570:
        preprocessed_recording = preprocessed_recording[:,:2570]
    elif preprocessed_recording.shape[1] < 2570:
        p = preprocessed_recording.shape[1]
        zer_prep = np.zeros([12,2570-p])
        preprocessed_recording = np.concatenate((preprocessed_recording,zer_prep),axis=1)
      
    recordings_array = np.array(recording)
    recordings_tensor = torch.from_numpy(recordings_array).float()
    return recordings_tensor




#Preprocess_ECG_signals
def preprocess(recording,fe,N,adc, baseline,leads):
    duration = 10
    target_fs = 250
    indices = list()
    available_leads=leads
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording=recording[indices,:]
    num_leads = len(leads)
    
    order=6
    bandpass = [0.5,120]
    nyq=fe/2
    
    record_tensor = []
    imputer = SimpleImputer(strategy='median')
    for lead_i in range(num_leads):
        #correct the voltage and baseline 
        ecg = (recording[lead_i,:]-baseline[lead_i])/adc[lead_i]

        #bandpass butterworth
        band=np.array(bandpass)/nyq
        b,a=signal.butter(order,band,'bandpass')
        ecg_filtered=signal.lfilter(b,a,ecg)

        #notch 60hz
        fn60=60#notch to remove
        bw=5#hz
        Q=fn60/bw#quality factor
        b_notch60,a_notch60=signal.iirnotch(fn60,Q,fs=fe)
        ecg_filtered=signal.lfilter(b_notch60,a_notch60,ecg_filtered)
        #notch 50hz
        fn50=50
        bw=5#hz
        Q=fn50/bw
        b_notch50,a_notch50=signal.iirnotch(fn50,Q,fs=fe)
        ecg_filtered=signal.lfilter(b_notch50,a_notch50,ecg_filtered)
        
        #resample to 250Hz
        ecg_filtered=processing.resample_sig(ecg_filtered,fe,target_fs)[0]
        
        #extract segment of the record
        #ecg_filtered = center_pad(ecg_filtered,duration,target_fs)
        ecg_filtered=imputer.fit_transform(ecg_filtered.reshape(-1,1)).reshape(1,-1)
        record_tensor.append(torch.tensor(ecg_filtered.copy()))
        # #deal with empty tensors
        # if ecg_filtered.shape[1]>0:
        #     record_tensor.append(torch.tensor(ecg_filtered.copy()))
        # else :
        #     record_tensor.append(torch.zeros((1,target_fs*duration)))


    norm_record=[]
    for j in range(num_leads):
            norm_record.append(stats.zscore(record_tensor[j]))


    record_tensor = torch.stack(record_tensor)
    print(record_tensor.shape)
    return np.array(norm_record)
    


def ecg_preprocessing(input_directory, tensors_directory,challenge_classes,equivalent_classes):

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)
    
    

    classes=get_classes_challenge(equivalent_classes,challenge_classes)
    classes_tensor = torch.from_numpy(np.array([float(classe) for classe in classes]))
    print("Saving classes done ...")

    num_classes = len(classes)
    num_files = len(header_files)
    recordings = list()
    headers = list()
    labels = list()
    
    # #preprocess the record
    # duration=10#seconds
    # target_fs = 250#Hz (incart is 257Hz)

                                                                                

    
    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        head = load_header(header_files[i])
        available_leads = get_leads(head)
        num_leads=len(available_leads)
        leads=available_leads
        fe,N= get_infos(header_files[i])
        
        adc, baseline = get_adc_gains(head, leads), get_baselines(head, leads)
        headers.append(header)

  


        preprocessed_recording=preprocess(recording, fe, N,adc, baseline,leads)
        if preprocessed_recording.shape[1] > 2500:
            preprocessed_recording = preprocessed_recording[:,:2500]
        elif preprocessed_recording.shape[1] < 2500:
            p = preprocessed_recording.shape[1]
            zer_prep = np.zeros([num_leads,2500-p])
            preprocessed_recording = np.concatenate((preprocessed_recording,zer_prep),axis=1)
        recordings.append(preprocessed_recording)
      
    recordings_array = np.concatenate(recordings)
    headers_array=np.array
    recordings_tensor = torch.from_numpy(recordings_array).float()
    len_recordings = len(recordings)
    
    #labels list creation 
    for i in range(len(headers)):
        header = headers[i]
        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(26)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    if(arr in challenge_classes):
                        for i in range(len(equivalent_classes)):
                                if(arr==equivalent_classes[i][0] or arr==equivalent_classes[i][1]):
                                    arr=equivalent_classes[i][1]
                    else:
                        arr=""
                    if arr!="":
                        class_index = classes.index(arr.rstrip()) # Only use first positive index
                        labels_act[class_index] = 1
                    else:
                        pass
 
        labels.append(labels_act)
    labels_array = np.array(labels)
 
    #print(labels_array)
    labels_tensor = torch.from_numpy(labels_array).float()
 
    #with open('/usr/users/gpusdi1/gpusdi1_8/Bureau/PFE/git/inserm/Classifier/Tensors/num_classes.pickle', 'wb') as f:
    #    pickle.dump(num_classes, f)
    return(recordings_array,recordings_tensor,labels_array,labels_tensor,classes,classes_tensor)




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
    
    #pour faire le preprocessing PCA
    
    #recordings_tensor = torch.load(tensors_directory + 'recordings_tensor.pt')
    #labels_tensor = torch.load(tensors_directory + 'labels_tensor.pt')
    #recordings_tensor = torch.load(os.path.join(tensors_directory, 'recordings_tensor.pt'))
    #labels_tensor = torch.load(os.path.join(tensors_directory, 'labels_tensor.pt'))
    print("Loading tensors done...")

    #pca_preprocess = pca_preprocess(n_recordings=5,n_components=1)#n_recordings nombre de recordings 
    #pca_analysis = pca_analysis(n_recordings=5,n_components=12) #pr plot les analyses
