#!/bin/bash

precursor_path="./precursor/SMD_PT_4800.pth"
dataset="PACS_on_Mario"

script="train_smos.py"
exp_name="multisource"


algorithms=(    
    "SMOS_JS"    
    "ERM"
    # "SagNet"
)

batch_sizes=(
    "16"
    # "32"
)

seeds=(
    "$RANDOM"
    "3"
    # "3407" 

)

# The hyperparam for grounding (JS-Divergence which is based on KL-Divergence)
ld_KLs=(
    0.15
    0.1
)

for bs in "${batch_sizes[@]}"
do

for algo in "${algorithms[@]}"
do

for seed in "${seeds[@]}"
do

for ld_KL in "${ld_KLs[@]}" 
do

python3 $script "train_${dataset}_algo_${algo}_ld_KL_${ld_KL}_batchsize_${bs}_seed_${seed}" --data_dir "/scratch/yluo97/data/dg/" \
    --work_dir "./output/" \
    --project_name "SMOS_${dataset}"\
    --algorithm "${algo}" \
    --dataset "${dataset}" \
    --ld_KL ${ld_KL}\
    --lr 3e-5 \
    --steps 5001 --model_save 5000 \
    --resnet_dropout 0.0 \
    --weight_decay 0.0 \
    --trial_seed "${seed}" --seed "${seed}" --batch_size "${bs}" --checkpoint_freq 200 \
    --miro_pretrained False --pretrained True \
    --smos_pre_featurizer_pretrained True --smos_pre_featurizer_path "${precursor_path}" 


sleep 30

done

done

done

done 