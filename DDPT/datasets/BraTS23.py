import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

""" IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
} """


@DATASET_REGISTRY.register()
class BraTS23(DatasetBase):

    dataset_dir = "BraTS23"

    def __init__(self, cfg):
        
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
    
        self.dataset_dir = os.path.join(root, self.dataset_dir)
       

        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        
        #.json file -> CONTAINS THE SPLIT OF THE DATA TRAIN,TEST,VAL
        #split_fewshot -> stores the pickled lists
        
        if(cfg.DATASET.TRAIN_PERCENT!=0):  #THIS IS RUN DURING TRAINING
          self.split_path = os.path.join(self.dataset_dir, "split_brats23.json")
          self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_train")
        else:                              #THIS IS RUN DURING TESTING
          self.split_path = os.path.join(self.dataset_dir, "split_brats23_full.json")
          self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_test")
        
        
        
        #making a split_fewshot directory
        mkdir_if_missing(self.split_fewshot_dir)
        
        
        #CREATING OR READING FROM SPLITS BASED USING FUNCTIONS FROM OxfordPets and DTD
        if os.path.exists(self.split_path):
            # train, val, test are lists of Datum objects based on the split produced
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else: 
            train, val, test = DTD.read_and_split_data(self.image_dir,cfg)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS

        #NOW WE CREATE BATCHES BASED ON THE NUMBER OF SHOTS REQUIRED
        if num_shots >= 1:
            seed = cfg.SEED
            #PATH NAME FOR THE FILE BASED ON THE SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            #IF THE FEW-SHOT DATA HAS ALREADY BEEN CREATED, WE JUST READ FROM THE PICKLE FILE
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            
            #ELSE WE CREATE THE BATCHES AND SAVE IT TO A PICKLE FILE
            else:
                #We use a function of the parent class to get a list of Datum object having the num_shots amount of objects from each class
                
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                #if num_shots=64, train=[128] (2 classes)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))  #would have a maximum of 4 shots in each list
                #val = [8]
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        

        #THE BELOW CODE HAS NO EFFECT IF WE HAVE subsample='all' (IF WE WANT ANY SUBSAMPLING IN THE LIST OF DATUM OBJECTS)
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
                            
        super().__init__(train_x=train, val=val, test=test)
