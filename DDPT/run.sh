j=1
MODEL=DPT
DATASET=BraTs20
DATADIR=
EVAL=True
CLASSIFY=True #This is not going to be used, will make a plan on this if we have to
TRAIN_CRAFT=True
EVAL_CRAFT=True
EPOCHS=60
MODEL_DIR=models/302/1024-1-302
NUM_SHOTS=1024
THRESHOLD=0
MEDIAN=True


# CPN is the length of CAVPT
# BOTTOMLIMIT is the layers CAVPT inserted into. e.g. 8 means 8-12 layers are equipped with CAVPT. 12 means 12 layers are equipped with CAVPT.1 means every layer are equipped with CAVPT.
# C is our general knowledge guidence epoch.
# ALPHA is loss balancing parameter.

python DDPT-model.py --root ../DATA/${DATADIR}  --seed $j --evaluate $EVAL --classify $CLASSIFY --train-craft $TRAIN_CRAFT --eval-craft $EVAL_CRAFT \
--trainer $MODEL \
--epochs $EPOCHS \
--threshold $THRESHOLD \
--median $MEDIAN \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--output-dir $MODEL_DIR \
--config-file ./configs/trainers/VPT/vit_b32_deep.yaml \
TRAINER.COOP.N_CTX 16 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS $NUM_SHOTS \
TRAINER.VPT.N_CTX 10 \
TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 \
TRAINER.SELECTED_COVPT.CPN 10 \
OPTIM.LR 0.01 \
OPTIM.MAX_EPOCH 60 \
PRETRAIN.C 30 \
TRAINER.ALPHA 0.3

