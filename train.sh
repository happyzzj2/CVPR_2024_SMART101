export CUDA_VISIBLE_DEVICES=0,1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# nohup python train.py --device="0,1,2,3" --batch_size 128 --lr 1e-5 --date="2024_5_28" --hyperp_f="VL_fusion_encoder_siglip_vit_L1_a" --cnt_layer 1 --num_epochs 10 --method="VL_encoder" > output_a.log 2>&1 &
python train.py --device="0,1" --batch_size 64 --lr 5e-6 --date="2024_6_2" --hyperp_f="VL_fusion_encoder_siglip_vit_deberta_L1_all" --cnt_layer 1 --num_epochs 7 --method="VL_encoder"
