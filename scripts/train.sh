CUDA_VISIBLE_DEVICES=2,3,4,5 nohup torchrun \
    --nproc_per_node=4 diy_main_ae.py \
    --accum_iter=2 \
    --model kl_d512_m512_l64  \
    --output_dir output/ae/kl_d512_m512_l64 \
    --data_path /data_new2/sz_zzz/Data/Teeth/RD_2 \
    --log_dir output/ae/kl_d512_m512_l64 \
    --num_workers 60 \
    --point_cloud_size 2048 \
    --batch_size 20 \
    --epochs 3000 \
    --kl_weight 0.001 \
    --warmup_epochs 80 > output.log 2>&1 &