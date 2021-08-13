##########################
##   Import Libraries   ##
##########################

import warnings
warnings.filterwarnings('ignore')

import numpy as np, os, sys, math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pickle
import tqdm
import copy
import new_model

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.module import _addindent
from sklearn.metrics import f1_score
import torchaudio

from helper_code import *
##############################
##   Some Utils Functions   ##
##############################

classes_names = ['maladie_'+str(i+1) for i in range(26)]

def order_databases(header_list, record_list, one_hot_labels, databases=["None"]):
    """
    "I" incart
    "E" georgia
    "HR" ptb xl
    "S" 
    "JS" chapman (the new one)
    "Q" cpsc complement
    "A" cpsc
    """
    headers,records,labels,weights= [],[],[],[]
    for idx in range(len(header_list)):
        for db in databases:
            #if ptb, disambiguate ptb from ningbo tag
            if db=="S":
                if not "JS" in header_list[idx]:
                    headers.append(header_list[idx])
                    records.append(record_list[idx])
                    labels.append(one_hot_labels[idx])
                    weights.append(torch.ones_like(one_hot_labels[idx]))
            #if "None", no db filtering
            #else, add the sample only if it is in the selected db
            elif db=="None" or db in header_list[idx]:
                headers.append(header_list[idx])
                records.append(record_list[idx])
                labels.append(one_hot_labels[idx])
                weights.append(torch.ones_like(one_hot_labels[idx]))
            
    
    return headers,records, labels, weights

def weight_db_classes(headers_list,weights_list,csv_dx_mapping_file,scored_classes):
    #dataframe from mapping csv file
    db_df = pd.read_csv(csv_dx_mapping_file)
    databases = ["I","Q","A","E","HR","S","JS"]
    #map the db tag of filenames to csv dbtag
    db2csv_map = {
        'I':'StPetersburg',
        'Q':'CPSC_Extra',
        'A':'CPSC',
        'E':'Georgia',
        'HR':'PTB_XL',
        'S':'PTB',
        'JS':'Ningbo'
        }
    
    weights = []
    for idx in range(len(headers_list)):
        #assign the db of the sample
        for db in databases:
            #disambiguate the database btw ptb and ningbo
            if db=="S":
                if db in headers_list[idx] and "JS" not in headers_list[idx]:
                    record_db = "S"
            else : 
                if db in headers_list[idx]:
                    record_db = db
        #get the distrib among scored classes
        db_distrib, db_distrib_total = [],[]
        for cls in scored_classes:
            db_num_pos = db_df[db_df['SNOMEDCTCode']==int(cls)][db2csv_map[record_db]].to_numpy()
            glob_num_pos = db_df[db_df['SNOMEDCTCode']==int(cls)]['Total'].to_numpy()
            db_distrib.append(db_num_pos)#db_df[db2csv_map[record_db]].to_numpy()
            db_distrib_total.append(glob_num_pos)
        db_distrib = np.array(db_distrib)
        db_distrib_total=np.array(db_distrib_total)
        
        #w = (db_distrib/db_distrib.sum()).reshape((len(db_distrib)))#p(class|database)
#        C = (db_distrib_total/db_distrib_total.sum()).reshape((len(db_distrib)))#p(classes)
        #C = (1/db_distrib_total.sum())
        #C = 1
        #w = w/C
        w = (db_distrib>0).astype(np.float).reshape((len(db_distrib)))
        weights.append(torch.from_numpy(w))
                                                
    return weights

def load_pretrained_model(model=None, pretrained_model_name=None, model_dir="./"):
    if pretrained_model_name == None :
        return
    else :
        model_path = os.path.join(model_dir,pretrained_model_name)
        print('retraining from a previous model')
        print('loading : ', model_path)
        model.load_state_dict(torch.load(model_path))
    return

def max_predict(preds):
    #predictions = torch.stack(preds)
    predictions = preds
    #vote
    return predictions.max(dim=0).values


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

        
def print_and_get_percentage_classes(labels_tensors):
    nb_elt_classes = labels_tensors.sum(axis=0)
    
    percentages_classes = nb_elt_classes/sum(nb_elt_classes)
    for i in range(len(percentages_classes)):
        if percentages_classes[i]==0:
           percentages_classes[i]=0.0000001
   
    #left = [i+1 for i in range(num_classes)] 
    #plt.bar(left, percentages_classes, width = 0.8, color = ['green']) 
    #plt.xlabel('classes') 
    #plt.ylabel('number_of_elements') 
    #plt.title('percentage_classes') 
    #plt.savefig('percentage_classes.png')
    return percentages_classes

def label_to_name():
    l=["atrial fibrillation,164889003","atrial flutter,164890007","bundle branch block,6374002","bradycardia,426627000","complete left bundle branch block,733534002","complete right bundle branch block,713427006","1st degree av block,270492004","incomplete right bundle branch block,713426002","left axis deviation,39732003","left anterior fascicular block,445118002","left bundle branch block,164909002","low qrs voltages,251146004","nonspecific intraventricular conduction disorder,698252002","sinus rhythm,426783006","premature atrial contraction,284470004","pacing rhythm,10370003","poor R wave Progression,365413008","premature ventricular contractions,427172004","prolonged pr interval,164947007","prolonged qt interval,111975006","qwave abnormal,164917005","right axis deviation,47665007","right bundle branch block,59118001","sinus arrhythmia,427393009","sinus bradycardia,426177001","sinus tachycardia,427084000","supraventricular premature beats,63593006","t wave abnormal,164934002","t wave inversion,59931005","ventricular premature beats,17338001"]
    d=[l[i].split(",") for i in range(30)]
    dic={i[1]: i[0] for i in d}
    return dic
def KeepNMax_with_indices(list1, N,list_class_name): 
    original_list = list1.copy()
    final_list = []
    for i in range(0, N):  
        max1 = 0 
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
        list1.remove(max1)
        final_list.append(max1)  
    for i, elt in enumerate(original_list):
        if elt not in final_list:
            original_list[i] = 0
    indices = [i for i in range(len(original_list)) if original_list[i] != 0]
    kept_classes_name = [list_class_name[k] for k in indices]
    weights = [original_list[i]  for i in range(len(original_list)) if original_list[i] != 0]
    return weights, indices,kept_classes_name


def KeepChallengeClasses(list_classes,challenge_list,list_weight,equivalent_classes):
    indices = []
    for disease in challenge_list:
        # ########################remove the if later !!!!!! ####################################""
        for i in range(len(equivalent_classes)):
            if(disease==equivalent_classes[i][0] or disease==equivalent_classes[i][1]):
                disease=equivalent_classes[i][1]
        if(str(int(disease)) in list_classes ):
            idx = list_classes.index(str(int(disease)))
            if(idx not in indices):
                indices.append(idx)
   # print("indice  "+str(len(indices)))
    # print("list_weight  "+str(len(list_weight)))
    weights = [list_weight[idx]  for idx in indices]
    # print("weight  "+str(len(weights)))
    x=len(weights)
    if(x<26):
        for i in range(x,26):
            weights.append(0.0000000000000000000001)
            indices.append(i)
    # print(len(weights))
    # print(len(indices))
    return weights, indices

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=6):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    dic=label_to_name()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_title("class is = {} ".format(dic[str(class_label)]), fontsize=fontsize)

#Note Sure if it's useful or not ?!
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def spectrogrammes(recordings):
    fs = 257
    return(torchaudio.transforms.Spectrogram(n_fft=1024,win_length=int(0.5*fs), hop_length=int(0.01*fs))(recordings))

def compute_class_weights(labels_tensor):
    wn =torch.nansum(labels_tensor,dim=0)
    pos_weight =(1-wn)/min(wn,1e-6)
    return wn, pos_weight

def multilead_predict(data_tensor, classifier, threshold, device="cpu", requires_sigmoid=True):
    """
    parameters : 
    data_tensor of size : Nbatch, Nchannels, Ntime
    classifier : the classifier applied to data_tensor
    threshold : either scalar or should match output dim of classifier
    (same num of classes as output labels)
    device : cpu or gpu (torch.device matching data_tensor and classifier)
    requires_sigmoid : in the case sigmoid is not applied within the classifier output layer
    (cf bcewithlogitloss)
    returns : 
    probabilities : probability vector from classifier's output
    labels : onehot labels
    clf_output : output of the classifier
    """
    
    sz = data_tensor.size()
    #src mask classifier
    mask = classifier.generate_square_subsequent_mask(sz[0])
    clf_output= classifier(data_tensor.view(sz[0]*sz[1],1,sz[2]).to(device),mask)
    print('classifier output size')
    print(clf_output.size())
    #if need sigmoid for probabilities
    if requires_sigmoid :
        clf_output = torch.sigmoid(clf_output)
        
    #reshape clf_output to data_tensor shape and take max along channel as final probability
    sz_clf = clf_output.size()
    clf_output = clf_output.view(sz[0],sz[1],sz_clf[1])
    #probabilities = torch.max(clf_output,dim=1).values.view(sz[0],sz_clf[1])
    probabilities = torch.mean(clf_output,dim=1).view(sz[0],sz_clf[1])

    labels = (probabilities>threshold).int()
    #labels = torch.ones_like(labels)
    return probabilities, labels

def multilead_compute(data_tensor,model):
    sz = data_tensor.size() #BatchxLeadsxTime
    model_output = model(data_tensor.view(sz[0]*sz[1],1,sz[2]).to(device))
    #transform (Batch*Leads)xTime-->(Batch*Leads)xEmbeddingxTime
    sz_output = model_output.size()
    model_output = model_output.view(sz[0],sz[1],sz_output[1],sz_output[2])#(Batch x Leads x Embedding x Time)
    return model_output
    
def train(model, loader, f_loss, optimizer,lr_scheduler ,device, weights, classes, normal_class,model_name,writer):
    threshold = 0.5
    preds, targets, loss = [],[],[]
    for i, (recordings, labels, batch_weights) in enumerate(tqdm.tqdm(loader)):

        recordings, labels, batch_weights = recordings.to(device), labels.to(device), batch_weights.to(device)
        
        
        src_mask = model.generate_square_subsequent_mask(recordings.size(0)).to(device)
        recordings = recordings.to(torch.float)
        outputs = model(recordings, src_mask)
        
        #outputs = model(recordings)
        train_loss = f_loss(outputs, labels, batch_weights)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # if lr_scheduler!=None:
        #     lr_scheduler.step()
        lr_scheduler.step()
        
        #stack tensors to compute training metrics
        loss.append(train_loss)
        pred_label = (torch.sigmoid(outputs)>threshold).int()
        preds.append(pred_label)
        targets.append(labels)
        

    running_loss = torch.tensor(loss).mean()
    preds = torch.cat(preds)
    preds = preds.to('cpu').detach().numpy()
    targets = torch.cat(targets).to('cpu').detach().numpy()

    # running_accuracy = compute_accuracy(targets,preds)
    # F1_Score , F1_Score_classe = compute_f_measure(targets,preds)

    #remove some metrics calculations to speed up the code
    running_accuracy,F1_Score , F1_Score_classe = 0,0,0
    
    challenge_metric = compute_challenge_metric(weights, targets, preds, classes, normal_class)
    print('training challenge metric', challenge_metric)
    return(running_loss,running_accuracy,F1_Score,challenge_metric)

def valid(model, loader, f_loss, device, weights, classes, normal_class,model_name, model_directory = 'model'):
    threshold = 0.5

    with torch.no_grad():
        preds, targets, loss = [],[],[]
        for i, (valid_recordings, valid_labels, valid_batch_weights) in enumerate(tqdm.tqdm(loader)):
            
            valid_recordings, valid_labels = valid_recordings.to(device),valid_labels.to(device)
  
            #predict outputs
            outputs,pred_label = multilead_predict(valid_recordings,model,threshold,device)
            val_loss = f_loss(outputs, valid_labels.to(device))
            #append the batch loss, predictions and targets
            loss.append(val_loss)
            preds.append(pred_label)
            targets.append(valid_labels)

            

        running_loss = torch.tensor(loss).mean()
        preds = torch.cat(preds)
        preds = preds.to('cpu').detach().numpy()
        targets = torch.cat(targets).to('cpu').detach().numpy()

    
    
    
    #challenge metrics
    
    # running_accuracy = compute_accuracy(targets, preds)
    # F1_Score_val , F1_Score_val_classe = compute_f_measure(targets,preds)
    # confusion_matrix = compute_confusion_matrices(targets,preds)
    #for challenge submission
    running_accuracy,F1_Score_val , F1_Score_val_classe = 0,0,0
    confusion_matrix = 0
    
    challenge_metric = compute_challenge_metric(weights, targets, preds, classes, normal_class)
    
    return(confusion_matrix, running_loss,running_accuracy,F1_Score_val,challenge_metric)

##   Summary writing functions   ##
###################################

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights.
        Return a tuple of the summary and the total number of parameters
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr, _ = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    tmpstr += '\n {} learnable parameters'.format(total_params)
    return tmpstr, total_params


def model_summary(model, optimizer, logdir):
    """ Generate and dump the summary of the model.
        model   a pytorch model object
    """
    model_summary, number_of_params = torch_summarize(model)
    # print("Summary:\n {}".format(model_summary))
    print("Total number of parameters: {}".format(number_of_params))

    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """
    Executed command
    ===============
    {}
    Dataset
    =======
    Model summary
    =============
    {}
    {} trainable parameters
    Optimizer
    ========
    {}
    """.format(" ".join(sys.argv),
            str(model).replace('\n','\n\t'),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            str(optimizer).replace('\n', '\n\t'))
    summary_file.write(summary_text)
    summary_file.close()

    return summary_text


class ModelCheckpoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.min_loss = None
    def update(self, loss, model):
        if (self.min_loss is None) or (loss < self.min_loss):
            print('saving a better model')
            torch.save(model.state_dict(),self.filepath)
            self.min_loss = loss
            


#######################################
## Some evaluation functions CiC2020 ##
#######################################

# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# For each set of equivalent classes, replace each class with the representative class for the set.
def saved_challenge_classes(classes,equivalents):
    l=[]
    
    print(len(classes))
    for i in range(len(classes)):
        arr=classes[i]
        tt=arr.split("|")
        for j in range(len(equivalents)):
            if(tt[0]==equivalents[j][0] or tt[0]==equivalents[j][1] ):
                arr=equivalents[j][1]
        if arr not in l:
            l.append(arr)
    print(l)
    return l
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
    
            if x in multiple_classes:
                classes[j] = multiple_classes[0][0] # Use the first class as the representative class.
    return classes

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    #print("num_rows =" + str(num_rows))
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))
    #print("num_cols =" + str(num_cols))
    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')
    #print("values shape"+ str(values.shape))
    return rows, cols, values


# Load weights.
def load_weights(weight_file):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weight_file)

    # Split the equivalent classes.
    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert(rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights

# Load labels from header/label files.
def load_labels(label_files, classes):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)

    # Iterate over the recordings.
    for i in range(num_recordings):
        header = load_header(label_files[i])
        y = set(get_labels(header))
        for j, x in enumerate(classes):

            if x & y:
                labels[i, j] = 1

    return labels



def class_weight(dx_csv_file, scored_classes, order = 1):
    classes_df = pd.read_csv(dx_csv_file)
    pos = []
    for cls in scored_classes :
        print(cls)
        num_pos = classes_df[classes_df['SNOMEDCTCode']==int(cls)]['Total'].to_numpy()

        print(num_pos)
        pos.append(num_pos)
    print(len(pos))
    pos = np.array(pos)
    print(pos)
    tot = pos.sum()
    print(tot)
    Wn = ((tot-pos)/pos)**order #weight=#negative samples / #positive_samples
    return torch.tensor(Wn).flatten()

####################################
##   Compute confusion matrices  and challenge metrics ##
####################################

def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)
    if not normalize:
        A = np.zeros((num_classes, 2, 2), dtype=int)
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    
    return A

def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure

# compute fmeasure per class
def compute_f_measure_perclass(labels,outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    f_measure_per_class = []
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            f_measure_per_class.append(f_measure[k])
        else:
            f_measure[k] = float('nan')
            f_measure_per_class.append(f_measure[k])

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure, f_measure_per_class

# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)
  

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
   
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


def calibrate(model,loader,weights, classes,normal_class, device):
    '''
    thresholds : list of thresholds
    '''
    #evaluation mode
    model.eval()
    values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

#    threshold = torch.zeros((1,len(classes))).to('cpu')
    threshold = 0.5*torch.ones((1,len(classes))).to('cpu')
    best_metric=-1
    #compute the classifier output probabilities
    with torch.no_grad():
        probs,targets,loss = [],[],[]
        for i,(valid_recordings,valid_labels,valid_batch_weights) in enumerate(tqdm.tqdm(loader)):
            sz = valid_recordings.size()#batchxnchannelsxtime
            mask = model.generate_square_subsequent_mask(sz[0])
            #get probabilities
            clf_output= torch.sigmoid((model(valid_recordings.view(sz[0]*sz[1],1,sz[2]).to(device),mask)))
            sz_clf = clf_output.size()
            #(batch*nchannels)xnclasses
            clf_output = clf_output.view(sz[0],sz[1],sz_clf[1])
            #(batchxnchannelsxnclasses)
            probabilities = torch.mean(clf_output,dim=1).view(sz[0],sz_clf[1])
            #add the preds and labels tensors
            probs.append(probabilities)
            targets.append(valid_labels)
        #stack probabilities and targets
        probabilities = torch.cat(probs).to('cpu')
        labels = torch.cat(targets).to('cpu').detach().numpy()
        for idx in range(threshold.size(-1)):#for each class 
            best_idx_thr= 0.5 #reinitialize best thr value
            #...??? best metric=-1
            for thr in values :#for each threshold value
                threshold[0,idx]=thr
                preds = (probabilities>threshold).int()
                preds = preds.to('cpu').detach().numpy()#numpy for computing challenge metric
           
                challenge_metric = compute_challenge_metric(weights,labels,preds,classes,normal_class)
                print('in utils.calibrate()')
                print('idx',idx,'val',thr,'challenge metric',challenge_metric)
                #if better challenge metric with the threshold
                if challenge_metric > best_metric:
                    best_idx_thr = thr #best threshold value for the specified index
                    best_metric = challenge_metric
            #set the best threshold at the idx
            threshold[0,idx] = best_idx_thr

    threshold = threshold.tolist() #threshold as list (for saving)
    print('best threshold, best metric: ',threshold, ' ',best_metric)
    return threshold
            
