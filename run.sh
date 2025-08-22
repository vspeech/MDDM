num_gpus=$(nvidia-smi -L | wc -l)
gpu_id=1
gpu_list=$(seq -s, $gpu_id $gpu_id)
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2024

num_workers=4
prefetch=10
data_type=raw
train_data=data/train_new.lst
#train_data=data/test.lst
#cv_data=data/cv.lst
train_config=conf/train_unet_hybrid.yaml
#tensorboard_dir=tensorboard

torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
            --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
xspeech/bin/train.py --config $train_config --data_type $data_type --train_data $train_data --num_workers ${num_workers} --prefetch ${prefetch} --pin_memory
