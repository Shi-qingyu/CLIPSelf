torchrun --nproc_per_node 4 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model ViT-L-14-336 --pretrained checkpoints/open_clip_pytorch_model.bin --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data data/coco/annotations/instances_train2017.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViTL14x336.npy --train-image-root data/coco/train2017 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/open_clip_pytorch_model.bin --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 24 --extract-type="v2" \
--name clipself_coco_6_save6_laion_vitl14___ --downsample-factor 14 --det-image-size 896 \
--alpha 0.95