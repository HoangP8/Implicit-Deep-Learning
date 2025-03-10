DEVICE=0 # GPU index
DATASET="spiky" # "spiky"
EPOCHS=100 # training epochs
HIDDEN_DIM=22 # hidden dimension of RNN model
IMPLICIT_HIDDEN_DIM=20
LR=5e-4 # training learning rate
KAPPA=0.99 # kappa for L-infty norm 
MAX_ITR=300 # max forward iterations
GRAD_MAX_ITR=300 # max backward iterations
TOL=3e-6 # forward tolerance
GRAD_TOL=3e-6 # backward tolerance
IS_LOW_RANK=True
RANK=2 # number of low-rank
SEED=0 # seed

python -m examples.idl_rnn.main \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --implicit_hidden_dim $IMPLICIT_HIDDEN_DIM \
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