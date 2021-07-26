import os
from datetime import datetime
import json

class ExpSupervisor():

    def __init__(self,logdir="./expt_logs/",
                 data_dir="/physionet/training_data/",
                 model_dir="/physionet/model/",
                 exp_file=None):
        
        #if no file specified, generate default experiment
        if exp_file!=None:
            return
        
        #my experiment tags and dir setup for saving
        self.expt_tag = self.generate_expt_tag()
        self.logdir = logdir

        #if the logdir does not already exist : create it
        if not os.path.isdir(logdir):
            os.mkdir(log_dir)
        #############my experiment metadata####################
        self.data_directory = data_dir
        self.model_directory = model_dir
        ############experiment parameters#####################
        self.model_name = 'trans'
        self.num_epochs=25
        self.batch_size=60
        self.kept_classes=26
        self.shuffle_dataset=True
        self.validation_split=0.25 #todo : check if unused

        self.leads =['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.list_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        self.equivalent_classes =[['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001'],['733534002', '164909002' ]]
        self.normal_class ={'426783006'}
        self.scoring_weights_csv ='weights.csv'
        self.dx_mapping_csv ='dx_mapping_scored.csv'

        self.n_folds = 5
        self.fold_idx=0

        self.ntoken=256
        self.emsize=512
        self.nhead=8# //
        self.nhid=128
        self.nlayers=6
        self.nchanels=1
        self.dropout=0.5

        #key of the corresponding criteria moduledicts in the team_code.py training_code
        self.model = 'se'
        self.criterion = 'bcelogits'
        self.optimizer = 'adam'
        
        #parameters for 'adam' optimizer
        self.adam_lr = 1e-4
        self.adam_weight_decay = 1e-8
        
        #parameters for 'noam' optimizer
        self.noam_d_model=512
        
        self.scheduler = 'step'
        #parameters for 'step' scheduler
        self.step_step_size = 33
        self.step_gamma = 0.45
        
        
        return
        

    def dump(self):
        """save the json file in self.logdir"""
        filename = os.path.join(self.logdir,self.expt_tag+".json")
        #convert the self.normal_class into list to make it serializable in json file
        self.normal_class=list(self.normal_class)
        #respectively in load convert self.normal_class into set
        with open(filename,'w') as fp:
            json.dump(self.__dict__,fp,indent=3)
        return
    
    def load(self,logfile):
        """logfile : .json file name"""
        with open(logfile) as json_file:
            data = json.load(json_file)
        self.__dict__.update(data)
        #convert normal class into set (respectively see dump(self)
        self.normal_class = set(self.normal_class)
        return

    def update(self,**kwargs):
        self.__dict__.update(kwargs)
        return
    
    def make_default_expt(self,logdir, data_dir, model_dir):
        """create a default experiment setup"""
        #my experiment tags and dir setup for saving
        self.expt_tag = self.generate_expt_tag()
        self.logdir = logdir

        #############my experiment metadata####################
        self.data_directory = data_dir
        self.model_directory = model_dir
        self.model_filename_fold = "/physionet/model/fold0.pt"
        self.mapping_csv_file = "/physionet/dx_mapping_scored.csv"

        self.n_fold_iter = 1
        #deprecated (to  change for official phase template)
        self.twelve_lead_model_filename = '12_lead_model.sav'
        self.six_lead_model_filename = '6_lead_model.sav'
        self.three_lead_model_filename = '3_lead_model.sav'
        self.two_lead_model_filename = '2_lead_model.sav'

        ############experiment parameters#####################
        self.train_batch_size = 32*6
        self.test_batch_size = 32*6
        self.num_epochs = 20
        self.tensorboard_logdir = os.path.join("/physionet/tensorboard_logs/",self.expt_tag)
        #architecture
        self.net = 'conv_transform'
        self.optimizer = 'adam'
        self.optimizer_weight_decay = None
        self.optimizer_Twarm=10
        self.optimizer_lrmin=5e-3
        self.optimizer_lrmax = 5e-1
        self.scheduler = 'step'
        self.scheduler_step_size=2
        self.scheduler_gamma=1

        return

    def generate_expt_tag(self):
        now = datetime.now()
        return now.strftime("%Y-%m-%d_%H%M%S_%f")

if __name__ == "__main__":
    print("test exp_supervision.py")
    exp = ExpSupervisor()
    exp.dump()
    print(exp.__dict__)
