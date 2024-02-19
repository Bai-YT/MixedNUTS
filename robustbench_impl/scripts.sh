# CIFAR-10
python run_robustbench.py --root_dir base_models --dataset_name cifar10 \
    --rob_model_name Peng2023Robust --std_model_arch rn152 \
    --map_type best --n_examples 10000 --batch_size_per_gpu 40

# CIFAR-100
python run_robustbench.py --root_dir base_models --dataset_name cifar100 \
    --rob_model_name Wang2023Better_WRN-70-16 --std_model_arch rn152 \
    --map_type best --n_examples 10000 --batch_size_per_gpu 40

# ImageNet
python run_robustbench.py --root_dir base_models --dataset_name imagenet \
    --rob_model_name Liu2023Comprehensive_Swin-L --std_model_arch convnext_v2-l_224 \
    --map_type best --n_examples 5000 --batch_size_per_gpu 20
