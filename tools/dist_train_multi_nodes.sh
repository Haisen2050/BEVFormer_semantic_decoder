cd /home/haisen/BEVFormer_segmentation_detection
conda activate bevformer-env
export TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 NCCL_SOCKET_IFNAME=enp36s0f0np0 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 EVAL_TMPDIR=/mnt/shared/eval_tmp

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1       # if it still fails, try 0 instead
export NCCL_SHM_DISABLE=0       # if /dev/shm is small or busy, try 1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_SOCKET_IFNAME=lo    # or your NIC, e.g. eth0 / enp3s0
export NCCL_DEBUG=INFO

# keep your spawn hook (sitecustomize.py)
export PYTHONPATH="$PWD:$PYTHONPATH"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr=192.168.0.249 \
    --master_port=29500 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    --launcher pytorch \
    --cfg-options model.pts_bbox_head.seg_head.type=FPN5Res18 data.workers_per_gpu=2 data.val_dataloader.workers_per_gpu=0 data.test_dataloader.workers_per_gpu=0 data.persistent_workers=True



# open a new terminal and run the following command on node2
ssh node2
cd /home/haisen/BEVFormer_segmentation_detection
export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 NCCL_SOCKET_IFNAME=enp4s0f0np0 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 EVAL_TMPDIR=/mnt/shared/eval_tmp
conda activate /home/haisen/conda_envs/bevformer-env
python -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=1 \
    --master_addr=192.168.0.249 \
    --master_port=29500 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    --launcher pytorch \
    --cfg-options model.pts_bbox_head.seg_head.type=FPN5Res18 data.workers_per_gpu=2 data.val_dataloader.workers_per_gpu=0 data.test_dataloader.workers_per_gpu=0 data.persistent_workers=True

# kill any stragglers
pkill -9 -f "tools/train.py|torch.distributed|torchrun" || true

# single-node, 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1

# tame NCCL for a workstation (PCIe, no IB)

./tools/dist_train.sh projects/configs/bevformer/bevformer_base_seg_det_150x150.py 2