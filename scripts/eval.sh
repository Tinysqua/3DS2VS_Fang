CUDA_VISIBLE_DEVICES=0 nohup python \
    diy_eval.py \
    --model kl_d512_m512_l64  \
    --data_path /data_new2/sz_zzz/Data/Teeth/RD_2 \
    --pth output/ae/kl_d512_m512_l64/checkpoint-1830.pth \
    --device cuda > eval.log 2>&1 &