torchrun \
    --nnodes=1 --nproc_per_node=8 training/main.py \
    --save-frequency 5 \
    --report-to "tensorboard" \
    --train-data train.jsonl \
    --val-data test_5k_mscoco_2014.jsonl \
    --val-data-flickr flickr_validation.csv \
    --csv-img-key "image" \
    --csv-caption-key "caption" \
    --csv-patch-key "jsonl" \
    --csv-separator "," \
    --dataset-type "jsonl" \
    --dataset-type-flickr "csv" \
    --warmup 2000 \
    --batch-size=256 \
    --lr 5e-4 \
    --epochs=100 \
    --workers=8 \
    --model Patch-ViT-B-16 \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --name output folder name \
    --cls-num $num \
    --activate True \
    --resume resume from a checkpoint.pt optional \