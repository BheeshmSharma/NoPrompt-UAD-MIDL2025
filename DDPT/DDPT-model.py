
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from yacs.config import CfgNode as CN

import trainers.DPT
import trainers.VLP
import trainers.VPT

import datasets.caltech101
import argparse

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    
    
    cfg.DATASET.SUBSAMPLE_CLASSES = 'all'

                


    #COMPLETELY NEW

         #THIS IS COMPLETELY NEW
    

    
    
    
    #END UPDATES

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX = 10 # VPT CTX num
    cfg.TRAINER.VPT.LN = False
    
    
    cfg.TRAINER.SELECTED_COVPT = CN()
    cfg.TRAINER.SELECTED_COVPT.CPN = 1 # SELECTED_COVPT CLASS_PROMPT_NUM
    
    cfg.TRAINER.TOPDOWN_SECOVPT = CN()
    cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT = 12
    cfg.TRAINER.TOPDOWN_SECOVPT.LR = 0.01
    
    
    cfg.PRETRAIN =CN()
    cfg.PRETRAIN.C=30
    
    cfg.TRAINER.ALPHA=1.0


    return cfg
    

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.epochs:
        cfg.OPTIM.MAX_EPOCH = args.epochs


    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


    # HAND CRAFT ARGUMENTS

    if(args.train_craft=="False"):
      cfg.TRAIN.HAND_CRAFT = False
    else:
      cfg.TRAIN.HAND_CRAFT = True

    if(args.eval_craft=="False"):
        cfg.EVAL.HAND_CRAFT = False 
    else:
        cfg.EVAL.HAND_CRAFT = True 
        
    # EVALUATION ARGUMENTS
    
    if(args.evaluate=="False"):
        cfg.EVAL.RUN=False
    else:
        cfg.EVAL.RUN=True

    if(args.classify=="False"):
        cfg.EVAL.CLASSIFY=False
    else:
        cfg.EVAL.CLASSIFY=True

    if(args.classify=='False'):
        cfg.EVAL.CLASSIFY=False
    else:
        cfg.EVAL.CLASSIFY=True
         

    cfg.EVAL.THRESHOLD=args.threshold  

    if(args.median=="False"):
        cfg.EVAL.MEDIAN=False
    else:
        cfg.EVAL.MEDIAN=True
        

        



def setup_cfg(args):

    cfg=get_cfg_default()
    cfg=extend_cfg(cfg)  # CHANGES WE WANTED TO BRING IN

    if args.dataset_config_file:
      cfg.merge_from_file(args.dataset_config_file)

      # 2. From the method config file
    if args.config_file:
      cfg.merge_from_file(args.config_file)
    
  # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    #now no more changes to the configs for the rest of the experiment
    #cfg.freeze()
    
    
    return cfg


    


def main(args):
    
    #print('ARGS',args)
    
    cfg=setup_cfg(args)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))

        set_random_seed(cfg.SEED)
    
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
      torch.backends.cudnn.benchmark = True
    
    train=False
    cond=True
    classify_metrics=False

    #trainer=build_trainer(cfg)
    #trainer.train()


    trainer=build_trainer(cfg)
    trainer.load_model(cfg.OUTPUT_DIR,epoch=cfg.OPTIM.MAX_EPOCH)  
    trainer.create_maps("map_location")

    '''

    if cfg.EVAL.RUN:
 
      cfg.DATASET.TRAIN_PERCENT=0
      cfg.DATASET.VAL_PERCENT=0
     
      # NEED TO REBUILD TRAINER TO GET ENTIRE DATASET INTO TESTING

      map_location=f'{cfg.DATASET.NAME}/{cfg.DATASET.NUM_SHOTS}/{cfg.EVAL.THRESHOLD}'
      trainer=build_trainer(cfg)
      #print("LOADING MODEL")
      trainer.load_model(cfg.OUTPUT_DIR,epoch=cfg.OPTIM.MAX_EPOCH)  
      #print("DONE LOADING")
      #print('CREATING MAPS')
      trainer.create_maps_eval(map_location)
      print('MAPS CREATED')
      #trainer.generate_metrics(map_location,threshold=cfg.EVAL.THRESHOLD,median=cfg.EVAL.MEDIAN)

    '''
    


if __name__=='__main__':
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--train-craft",
        type=str,
        default="False",
        help="If the user wants to use Hand Crafted Prompts during Training",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default="False",
        help="If the user wants to run evaluation additional to the training",
    )
    parser.add_argument(
        "--classify",
        type=str,
        default="False",
        help="If the user wants to let the model classify before evaluating",
    )
    parser.add_argument(
        "--median",
        type=str,
        default="False",
        help="If the user wants to apply median filter during evaluation",
    )
    parser.add_argument(
        "--eval-craft",
        type=str,
        default="False",
        help="If the user wants to use Hand Crafted Prompts during Evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--threshold", type=int, default=-1, help="Threshold to apply during evaluation"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of epochs for training"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    

    main(args)
    
    


 
    
    #SETING UP THE SPLIT WHILE MAKING A NEW DATASET
    # cfg.DATASET.NAME='MSD'
    # trainer=build_trainer(cfg)
    
    
    #TRAINING A MODEL
    
    '''
    
    cfg.DATASET.NAME='BraTS23'
    cfg.DATASET.NUM_SHOTS= 10
    cfg.OPTIM.MAX_EPOCH= 5
    cfg.OUTPUT_DIR=f'models/23/test'
    trainer=build_trainer(cfg)
    trainer.train()
    
    '''
    
    #CREATING MAPS
    
    
    '''
    cfg.DATASET.TRAIN_PERCENT=0
    cfg.DATASET.VAL_PERCENT=0
    cfg.DATASET.NAME=f'BraTS23'
    model_dir=f'models/23/test'
    load_epoch= 5
    trainer=build_trainer(cfg)
    trainer.load_model(model_dir,epoch=load_epoch)
    trainer.create_maps(f'23/test/0')
    '''
    
    #CHECKING METRIC
    '''
    
    cfg.DATASET.TRAIN_PERCENT=0
    cfg.DATASET.VAL_PERCENT=0
    cfg.DATASET.NAME=f'BraTS23'
    model_dir=f'models/23/test'
    load_epoch= 5
    trainer=build_trainer(cfg)
    trainer.load_model(model_dir,epoch=load_epoch)
    #trainer.create_maps(f'23/test/0')
    trainer.generate_metric(f'23/test/0',0,False)
    '''
    
    
    ## ARCHIVE CODE
    
    #TRAINING USING FOR LOOP
    
    '''
    cfg.DATASET.NAME='Brain302'
    shots=[64,256,1024]
    for i in shots:
      print('HITTING A ', i ,' SHOT')
      cfg.DATASET.NUM_SHOTS= i
      cfg.OPTIM.MAX_EPOCH= 60
      cfg.OUTPUT_DIR=f'models/302/{i}-0-302'
      trainer=build_trainer(cfg)
      trainer.train()
    '''
    
    
    #TRAINING THE FOLDS
    
    '''
    cfg.DATASET.NAME='Brain302'
    cfg.OUTPUT_DIR=f'models/302/10000-100-1-401-F1'
    cfg.DATASET.NUM_SHOTS= 10000
    cfg.OPTIM.MAX_EPOCH= 100
    trainer=build_trainer(cfg)
    trainer.train()
    '''
    
    
    
    
    
    #CREATING METRICS
    
    '''
    
    cfg.DATASET.TRAIN_PERCENT=0
    cfg.DATASET.VAL_PERCENT=0
    
    
    models=[401]
    
    for j in models:
    
      shots=[256]
      
      for i in shots:
        cfg.DATASET.NAME=f'BraTS23'
        model_dir=f'models/{j}/{i}-0-{j}'
        load_epoch= 60
        trainer=build_trainer(cfg)
        trainer.load_model(model_dir,epoch=load_epoch)
        #trainer.create_maps(f'{j}/{i}/0')
        
        #trainer.generate_metric(f'{j}/{i}/0',400,False)
        
    '''
    
    # #BRATS21 TRAINED MODEL
    
    # folds=['F1','F2','F3','F4','F5']
    
    # for i in folds:
    #   cfg.DATASET.NAME=f'BraTS23'
    #   model_dir=f'models/401/10000-100-0-401-{i}'
    #   load_epoch= 100
    #   trainer=build_trainer(cfg)
    #   trainer.load_model(model_dir,epoch=load_epoch)
    #   trainer.create_maps(f'Tr21_Ts23/fold_{i}/0')
    #   trainer.generate_metric(f'Tr21_Ts23/fold_{i}/0',0,False)
      
    # BRATS20 TRAINED MODEL
    
    '''
    
    folds=['F1','F2','F3','F4','F5']
    
    for i in folds:
      cfg.DATASET.NAME=f'MSD'
      model_dir=f'models/302/10000-100-0-302-{i}'
      load_epoch= 100
      trainer=build_trainer(cfg)
      trainer.load_model(model_dir,epoch=load_epoch)
      trainer.create_maps(f'Tr20_Tsmsd/fold_{i}/0')
      trainer.generate_metric(f'Tr20_Tsmsd/fold_{i}/0',0,False)
    '''
    
    
    #FOR THE L1,H1 THING
    '''
    folds=['F1']
    
    
    #folds=['F2','F3','F4','F5']
    
    thresholds=[0]
    
    cfg.DATASET.NAME='Brain401'
    trainer=build_trainer(cfg)
   
    print(len(trainer.test_loader))
    
    
    for fold in folds:
      model_dir=f'./models/302/10000-100-1-302-{fold}'
      load_epoch=100
      trainer.load_model(model_dir,epoch=load_epoch)
      #trainer.create_maps(f'302-401/10000-100-1-302-{fold}/3')
    
      for i in thresholds:
        trainer.generate_metric(f'302-401/10000-100-1-302-{fold}/4',i,False)
    '''
    
    
    #MEDIAN FILTER VALA
    '''
    folds=['F1','F2','F3','F4']
    thresholds=[0,200,400]
    cfg.DATASET.NAME='Brain302'
    trainer=build_trainer(cfg)
    print(len(trainer.test_loader))
    
    for fold in folds:
      model_dir=f'./models/401/10000-100-0-401-{fold}'
      load_epoch=100
      trainer.load_model(model_dir,epoch=load_epoch)
      #trainer.create_maps(f'401-302/10000-100-0-401-{fold}/1')
      trainer.generate_metric(f'401-302/10000-100-0-401-{fold}/1',0,True)
    '''
    
    #ABLATION STUDY TRAINING
    '''
    
    alphas=[0.7]
    
    c=1
    
    for alpha in alphas:
    
      cfg.DATASET.NAME='Brain302'
      cfg.OUTPUT_DIR=f'models/302/10000-100-1-302-F1-A6'
      cfg.DATASET.NUM_SHOTS= 10000
      cfg.OPTIM.MAX_EPOCH= 100
      cfg.TRAINER.ALPHA=alpha
      trainer=build_trainer(cfg)
      trainer.train()
      c+=1
    
    '''
    #ABLATION STUDY MASKS
    '''
    folds=['F1']
    
    thresholds=[0]
    types=['A5','A6']
    
    cfg.DATASET.NAME='Brain401'
    trainer=build_trainer(cfg)
   
    print(len(trainer.test_loader))
    
    
    for Type in types:
      model_dir=f'./models/302/10000-100-1-302-F1-{Type}'
      load_epoch=100
      trainer.load_model(model_dir,epoch=load_epoch)
      trainer.create_maps(f'302-401/10000-100-1-302-F1-{Type}/1')
      for i in thresholds:
        trainer.generate_metric(f'302-401/10000-100-1-302-F1-{Type}/1',i, False)
    '''
    
    
    
    '''
    shots=['64','256','1024','10000-100']
    #shots=['10000-100']
    
    #cat=[3,4]
    
    trainer=build_trainer(cfg)
    for i in shots:
      model_dir=f'./models/401/{i}-0-401'
      if i=='10000-100':
        load_epoch=100
      else:
        load_epoch=60
      trainer.load_model(model_dir, epoch=load_epoch)
      trainer.create_maps(f'401/{i}/2') 
    '''
       
    '''
       
    #shots=['64','256','1024','10000-100']
    shots=['10000-100']
    #thresholds=[200,400]
    thresholds=[0]
    cat=[3,4]
    #cat=[2]
    trainer=build_trainer(cfg)
    for i in shots:
      for j in thresholds:
        for c in cat:
          model_dir=f'./models/401/{i}-1-401'
          if i=='10000-100':
            load_epoch=100
          else:
            load_epoch=60
          trainer.load_model(model_dir, epoch=load_epoch)
          trainer.generate_metric(f'401/{i}/{c}',j)
    '''
    
    '''
    
    cfg.DATASET.NAME='Brain501'
    
    folds=['F1','F2','F3','F4','F5']

    trainer=build_trainer(cfg)
    
    
    for fold in folds:
    
      path=f'401-501/10000-100-0-401-{fold}/1'
    
      print('FOLD',fold)
    
      model_dir=f'./models/401/10000-100-0-401-{fold}'
      load_epoch=100
      trainer.load_model(model_dir,epoch=load_epoch)
      
      trainer.check(fold,path)
    
    
    '''
