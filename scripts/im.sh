# Example for MNIST and CIFAR-10
python -m examples.im.main --dataset mnist --hidden_dim 100 --lr 5e-3
python -m examples.im.main --dataset mnist --hidden_dim 100 --lr 5e-3 --low_rank True

python -m examples.im.main --dataset cifar10 --hidden_dim 300 --lr 5e-4
python -m examples.im.main --dataset cifar10 --hidden_dim 300 --lr 5e-4 --low_rank True --rank 2


# Example for Netflix and synthetic Spiky dataset
python -m examples.im_rnn.main --dataset spiky --hidden_dim 22 --lr 5e-4 --low_rank True --implicit_hidden_dim 20
python -m examples.im_rnn.main --dataset netflix --hidden_dim 22 --lr 5e-4 --low_rank True --implicit_hidden_dim 20