DEVICE=0 # GPU index
DATASET="mnist" # "mnist" or "cifar10"
EPOCHS=10 # training epochs
HIDDEN_DIM=100 # hidden dimension of Implicit model
LR=5e-3 # training learning rate
KAPPA=0.99 # kappa for L-infty norm 
BATCH_SIZE=64 # training and testing batch size
MAX_ITR=300 # max forward iterations
GRAD_MAX_ITR=300 # max backward iterations
TOL=3e-6 # forward tolerance
GRAD_TOL=3e-6 # backward tolerance
IS_LOW_RANK=True
RANK=2 # number of low-rank
SEED=0 # seed

python -m examples.idl.main \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR \
    --device $DEVICE \
    --kappa $KAPPA \
    --mitr $MAX_ITR \
    --grad_mitr $GRAD_MAX_ITR \
    --tol $TOL \
    --grad_tol $GRAD_TOL \
    --seed $SEED \
    --is_low_rank $IS_LOW_RANK \
    --rank $RANK