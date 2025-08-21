CUDA_VISIBLE_DEVICES=4 python extract_multiple_model_slide_embedding.py \
    --checkpoint  /mnt/radonc-li02/private/luoxd96/omnipath/ELF/checkpoint/checkpoint_best.pth \
    --dataset ebrains \
    --feature-models virchow2 \
    --output-path /mnt/radonc-li02/private/luoxd96/omnipath/ELF/evaluation/ebrains/virchow2_elf &


CUDA_VISIBLE_DEVICES=5 python extract_multiple_model_slide_embedding.py \
    --checkpoint  /mnt/radonc-li02/private/luoxd96/omnipath/ELF/checkpoint/checkpoint_best.pth \
    --dataset ebrains \
    --feature-models uni \
    --output-path /mnt/radonc-li02/private/luoxd96/omnipath/ELF/evaluation/ebrains/uni_elf &


CUDA_VISIBLE_DEVICES=6 python extract_multiple_model_slide_embedding.py \
    --checkpoint  /mnt/radonc-li02/private/luoxd96/omnipath/ELF/checkpoint/checkpoint_best.pth \
    --dataset ebrains \
    --feature-models gigapath \
    --output-path /mnt/radonc-li02/private/luoxd96/omnipath/ELF/evaluation/ebrains/gigapath_elf &

CUDA_VISIBLE_DEVICES=7 python extract_multiple_model_slide_embedding.py \
    --checkpoint  /mnt/radonc-li02/private/luoxd96/omnipath/ELF/checkpoint/checkpoint_best.pth \
    --dataset ebrains \
    --feature-models h0 \
    --output-path /mnt/radonc-li02/private/luoxd96/omnipath/ELF/evaluation/ebrains/h0_elf &

CUDA_VISIBLE_DEVICES=7 python extract_multiple_model_slide_embedding.py \
    --checkpoint  /mnt/radonc-li02/private/luoxd96/omnipath/ELF/checkpoint/checkpoint_best.pth \
    --dataset ebrains \
    --feature-models conch_v1_5 \
    --output-path /mnt/radonc-li02/private/luoxd96/omnipath/ELF/evaluation/ebrains/conch_v1_5_elf


