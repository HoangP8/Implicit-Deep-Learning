DEVICE=0 # GPU index
DATASET="tinyshakespeare" # "tinyshakespeare" or "tinystories" or "wikitext"
LR=1e-4 # training learning rate
IS_LOW_RANK=True
RANK=2 # number of low-rank
SEED=0 # seed

python -m examples.idl_attention.main \
    --dataset $DATASET \
    --lr $LR \
    --device $DEVICE \
    --seed $SEED \
    --is_low_rank $IS_LOW_RANK \
    --rank $RANK