#!/usr/bin/env python

# Do *not* edit this script.

import sys
from team_code import training_code

if __name__ == '__main__':
    l=["WFDB_ChapmanShaoxing","WFDB_CPSC2018","WFDB_CPSC2018_2","WFDB_Ga","WFDB_Ningbo","WFDB_PTB","WFDB_PTBXL","WFDB_StPetersburg"]
    data_directory = sys.argv[1]#give the path of the parent directory where all the dataset exists
    model_directory = sys.argv[2]
    for i in l:
        training_code(data_directory+i, model_directory) ### Implement this function!
        print("Done training on the Dataset :"+data_directory)

    print('Done.')
