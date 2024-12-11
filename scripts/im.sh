# Example for MNIST and CIFAR-10
python -m examples.im.main --dataset mnist --hidden_dim 100 --lr 5e-3
python -m examples.im.main --dataset mnist --hidden_dim 100 --lr 5e-3 --is_low_rank True

python -m examples.im.main --dataset cifar10 --hidden_dim 300 --lr 5e-4
python -m examples.im.main --dataset cifar10 --hidden_dim 300 --lr 5e-4 --is_low_rank True --rank 2