import random
import numpy as np
import matplotlib.pyplot as plt

def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def stratified_k_fold(one_hot_mat, n_folds, ignore_nulls=False ,shuffle=True):
    """create n_folds folds that are stratified
    parameters : 
    one_hot_mat : matrix of the onehot encoded classes (each line is a data sample, 
    each column is a class)
    n_folds : number of folds to create
    ignore_nulls = False : if set to true, samples that are full zeros are not distributed : todo : implement (the if part)
    shuffle=True : if set to true, shuffle the folds as they are created
    
    return : 
    folds : a list of lists : each list within folds is a list of indices
    that corresponds to the samples distributed to this fold
    warning : one fold corresponds to a test fold. for the training fold
    take the complement of the test fold. 
    """
    distrib = one_hot_mat.copy()
    full_distrib = np.sum(one_hot_mat, axis=0)
    col_distributed = np.full(one_hot_mat.shape[1],np.NaN)
    #initialize folds
    folds = []
    null_fold = []
    for k in range(n_folds):
        folds.append([])
    if not ignore_nulls:
        print('include null ecg diags')
        
        #distribute folds that have a full zero line
        is_zero_line = [np.all(distrib[i,:]==0) for i in range(distrib.shape[0])]
        print(np.sum(is_zero_line),'fully zeros')
        iteration=0
        for i in range(len(is_zero_line)):
            if is_zero_line[i]:
                fold_idx=iteration%n_folds
                iteration+=1
                folds[fold_idx].append(i)
                null_fold.append(i)
    else :
        print('ignoring null ecg diags')
    #as long as not all folds are not distributed
    iteration=0
    while(np.isnan(np.sum(col_distributed))):
        iteration +=1 
    # for k in range(5):
        #take the feature with least samples
        feat_to_distribute=np.nanargmin(full_distrib)

        # print(feat_to_distribute)
        #distribute the indices along with that feature
        samp_to_distribute = np.where(distrib[:,feat_to_distribute]>0)
        for i in range(len(samp_to_distribute[0])):
            fold_idx = i%n_folds
            #fold_idx = iteration%n_folds
            #fold_idx = np.random.randint(0,n_folds)
            folds[fold_idx].append(samp_to_distribute[0][i])
            full_distrib-=distrib[samp_to_distribute[0][i],:]
            distrib[samp_to_distribute,:]*=0

            #if desired, we shuffle the fold 
            if shuffle : 
                random.shuffle(folds[fold_idx])
        #remove the feature from the nan list
        col_distributed[feat_to_distribute]=0
        full_distrib[feat_to_distribute]=np.nan
       
      
    return folds

        
if __name__=="__main__":
    
    one_hot_mat = rand_bin_array(50000,60000)
    one_hot_mat = one_hot_mat.reshape((10000,6))
    distrib = np.sum(one_hot_mat,axis=0)/np.sum(one_hot_mat)
    plt.matshow(distrib.reshape((1,len(distrib))))
    fold_list = stratified_k_fold(one_hot_mat,4)

    folds = [one_hot_mat[i,:] for i in fold_list]
    fold_distrib = [np.sum(i,axis=0)/np.sum(i) for i in folds]
    for i in fold_distrib : 
        plt.matshow(i.reshape(1,len(i)))
    print(one_hot_mat.shape)
    print([len(i) for i in folds])
    print (np.sum(np.array([len(i)for i in folds])))
