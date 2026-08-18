pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html
python setup.py build_ext --inplace 
# OccScanNet
# vggt or dav2
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/mono_vggt_bin16_release_custom.py \
    --work-dir /c20250502/wangyushen/Outputs/gpocc/vggt_bin16/train1

# OccScannet dav2
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/mono_dpt_bin16_release_custom.py \
    --work-dir /c20250502/wangyushen/Outputs/gpocc/mono_dpt_bin16/train1


# EmbodiedOcc-ScanNet GPOcc仅推理 vggt
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_embodied.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/embodied_vggt_custom.py \
    --work-dir /c20250502/wangyushen/Outputs/gpocc/embodiedocc_vggt_bin16/val1 \
    --evaluate \

# EmbodiedOcc-ScanNet GPOcc仅推理 dav2
PYTHONPATH=`pwd`/src:`pwd`/src/gpocc/Depth-Anything-V2/metric_depth:$PYTHONPATH \
python /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/scripts/train_embodied.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc/config/embodied_dpt_custom.py \
    --work-dir /c20250502/wangyushen/Outputs/gpocc/embodiedocc_dav2_bin16/val1 \
    --evaluate \
# ==================================================#
# 火山引擎 多卡训练 OccScannet
# vggt
cd /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
bash scripts/train_vggt_bin16.sh

# dav2
cd /vepfs-mlp2/c20250502/haoce/wangyushen/GPOcc
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
bash scripts/train_dav2_bin16.sh