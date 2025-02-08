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
class MSLUB(DatasetBase):

    dataset_dir = "MSLUB"

    def __init__(self, cfg):
        
        
      
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        #now just append a lil to each one of these

        #1) the image directory
        self.image_dir = os.path.join(self.dataset_dir, "images")
        #2) file which has the image info
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_brainFULL.json")
        #3) some new folder to store a pickle object of the few shot data were gonna use (based on seed)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        
        #making a split_fewshot directory
        mkdir_if_missing(self.split_fewshot_dir)

        #train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        
        
        
        '''RUN THIS PART IF YOU NEED TO MAKE THE SPLIT'''
        #train, val, test = DTD.read_and_split_data(self.image_dir)
        #OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        
        '''RUN THIS IF THE SPLIT IS ALREADY MADE'''
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        #print('TRAIN',len(train))

        #Once we have the train, val and test -> we feed it back into the DatasetBase Class
        
        if os.path.exists(self.split_path):
            print('yes')
            #POINT HERE BEING THAT SINCE A SIMILAR FUNCTION IS ALREADY BEING USED IN OXFORDPETS, THEYRE JUST USING IT BACK HERE
            # SO TRAIN, VAL AND TEST ARE GOING TO BE LISTS OF DATUM OBJECTS (HAVING IMAGE PATH, LABEL AND CLASSNAME FOR EACH IMAGE)
            #train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        #else: 
            #train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            #OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS

        #IN CASE SHOTS DO EXIST
        if num_shots >= 1:
            seed = cfg.SEED
            #PATH NAME FOR THE FILE BASED ON THE SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                #IN CASE THE SEED HAS ALREADY BEEN REACHED BEFORE
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                #THIS IS NOW A FUNCTION OF THE Dataset class -> 
                #Point here being is that we get a list of Datum object having the num shots amount of objects from each class
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                #making a dict of this and then saving it to this split_fewshot directory
                #(this is what caused the issue back then when trying to run on MobaXterm)
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        #again another common function from OxfordPets
        #but for the cfg we passed in, it basically just returns the same train,val and test outside [MIGHT HOWEVER WANNA TRY TESTING OUT WITH THIS]
        super().__init__(train_x=train, val=val, test=test)
        

#So to conclude off, everything will pretty much go smooth with the same code itself, would just have to 
        #1) create a new database in DATA
        #2) but images inside one folder
        #3) make a split.json file in similar format to the one in caltech
        #4) save it into the registry
        #5) make changes along the cfg path to access this dataset now
