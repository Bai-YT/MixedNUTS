# Calculate margin for the CIFAR-10 SOTA models
for image_type in clean aa; do
    python calc_margins.py --dataset_name cifar10 --model_name Peng2023Robust \
        --image_type $image_type --batch_size 50 --n_examples 1000
done
# Determine the s, p, c, alpha values for the CIFAR-10 models
python calc_spca.py --dataset_name cifar10 --rob_model_name Peng2023Robust \
    --target_rob_beta 98.5 --nonlin_type gelu --batch_size 100
# Run RobustBench on Mixed Classifier
python run_robustbench.py --root_dir base_models --dataset_name cifar10 \
    --rob_model_name Peng2023Robust --std_model_arch rn152 --map_type identity \
    --adaptive --n_examples 10000 --batch_size_per_gpu 40
python run_robustbench.py --root_dir base_models --dataset_name cifar10 \
    --rob_model_name Peng2023Robust --std_model_arch rn152 --map_type best \
    --adaptive --n_examples 10000 --batch_size_per_gpu 40 --disable_nonlin_for_grad


# Calculate margin for the CIFAR-100 SOTA models
for image_type in clean aa; do
    python calc_margins.py --dataset_name cifar100 --model_name Wang2023Better_WRN-70-16 \
        --image_type $image_type --batch_size 50 --n_examples 1000
done
# Determine the s, p, c, alpha values for the CIFAR-100 SOTA models
python calc_spca.py --dataset_name cifar100 --rob_model_name Wang2023Better_WRN-70-16 \
    --target_rob_beta 98.5 --nonlin_type gelu --batch_size 100
# Run RobustBench on Mixed Classifier
python run_robustbench.py --root_dir base_models --dataset_name cifar100 \
    --rob_model_name Wang2023Better_WRN-70-16 --std_model_arch rn152 --map_type identity \
    --adaptive --n_examples 10000 --batch_size_per_gpu 40
python run_robustbench.py --root_dir base_models --dataset_name cifar100 \
    --rob_model_name Wang2023Better_WRN-70-16 --std_model_arch rn152 --map_type best \
    --adaptive --n_examples 10000 --batch_size_per_gpu 40 --disable_nonlin_for_grad


# Calculate margin for the ImageNet-1k SOTA models
for image_type in clean aa; do
    python calc_margins.py --dataset_name imagenet --model_name Liu2023Comprehensive_Swin-L \
        --pair --image_type $image_type --batch_size 25 --n_examples 1000
done
# Determine the s, p, c, alpha values for the ImageNet-1k SOTA models
python calc_spca.py --dataset_name imagenet --rob_model_name Liu2023Comprehensive_Swin-L \
    --target_rob_beta 99.0 --nonlin_type gelu --batch_size 50
# Run RobustBench on Mixed Classifier
python run_robustbench.py --root_dir base_models --dataset_name imagenet \
    --rob_model_name Liu2023Comprehensive_Swin-L --std_model_arch convnext_v2-l_224 --map_type identity \
    --adaptive --n_examples 5000 --batch_size_per_gpu 20
python run_robustbench.py --root_dir base_models --dataset_name imagenet \
    --rob_model_name Liu2023Comprehensive_Swin-L --std_model_arch convnext_v2-l_224 --map_type best \
    --adaptive --n_examples 5000 --batch_size_per_gpu 20 --disable_nonlin_for_grad
