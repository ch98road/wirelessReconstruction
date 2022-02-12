
# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 test_multi.py
CUDA_VISIBLE_DEVICES=0,2,4 python -m torch.distributed.launch --nproc_per_node=3 Model_train_office_multi.py
CUDA_VISIBLE_DEVICES=0,2,4 python -m torch.distributed.launch --nproc_per_node=3 Model_train_CRBCA_LQ_multi_LayerNorm.py
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 Model_train_CRBCA_LQ_multi_LayerNorm_L.py