
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --node_rank=$MLP_ROLE_INDEX \
    --master_port=$MLP_WORKER_0_PORT \
    --nnodes=$MLP_WORKER_NUM \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/mono_dpt_bin16_release_custom_experiment.py \
    --work-dir /c20250502/wangyushen/Outputs/gpocc/dav2_bin16/train
