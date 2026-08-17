pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html

# vggt or dav2
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/mono_vggt_bin16_release_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/outputs/train1


# 火山 多卡训练
cd /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
bash scripts/train_vggt_bin16.sh