# Attention
python -m examples.im_attention_2.main --is_low_rank --max_iters 1 --fixed_point_iter 1 --dataset tinyshakespeare \
--device 0 --rank 3 --attention_version lipschitz

python -m examples.im_attention_2.main --max_iters 1 --fixed_point_iter 1 --dataset wikitext \
--device 0