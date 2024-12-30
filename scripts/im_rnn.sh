# Example for Netflix and synthetic Spiky dataset
python -m examples.im_rnn.main --dataset spiky --hidden_dim 22 --lr 5e-4 --is_low_rank True --implicit_hidden_dim 20
python -m examples.im_rnn.main --dataset netflix --hidden_dim 22 --lr 5e-4 --is_low_rank True --implicit_hidden_dim 20