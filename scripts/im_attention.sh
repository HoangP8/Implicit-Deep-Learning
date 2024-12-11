# Attention
python -m examples.im_attention.main --is_low_rank --max_iters 1 --fixed_point_iter 1 --dataset tinyshakespeare \
--device 0 --rank 3 --init_implicit_from_explicit --attention_version lipschitz

python -m examples.im_attention.main --max_iters 1 --fixed_point_iter 1 --dataset wikitext \
--device 0 --init_implicit_from_explicit 