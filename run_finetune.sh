#!/bin/bash


    
    python3 /home/linfeng/toxfm/run_finetune_with_folds.py \
        --base_dir /home/linfeng/toxfm/dataset_splits/heart_no_overlap_42 \
        --pretrained_model_path /home/linfeng/pretrain-gnns-2/multimodal_four71_graphtrans_model.pth \
        --gene_list_file /home/linfeng/toxfm/chem/gene_list1.txt \
        --modalities    mol \
        --epochs 6 \
        --batch_size 32 \
        --learning_rate 4e-5 \
        --device cuda:1 \
        --output_dir /home/linfeng/toxfm/finetune_heart_new/mol


