cd /home/haisen/BEVFormer_segmentation_detection
conda activate bevformer-env
export NCCL_SOCKET_IFNAME=enp36s0f0np0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr=192.168.0.249 \
    --master_port=29500 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    --launcher pytorch

ssh node2
cd /home/haisen/BEVFormer_segmentation_detection
export NCCL_SOCKET_IFNAME=enp4s0f0np0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
conda activate /home/haisen/conda_envs/bevformer-env
python -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=1 \
    --master_addr=192.168.0.249 \
    --master_port=29500 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    --launcher pytorch
