
python ECE/calculate_ece.py --M T1w --is_teacher --D CIFAR100_imb100
python ECE/calculate_ece.py --M T2w --is_teacher --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s --is_teacher --D CIFAR100_imb100
python ECE/calculate_ece.py --M T2s --is_teacher --D CIFAR100_imb100


# python ECE/calculate_ece.py --M only_Sw --D CIFAR100_imb100
# python ECE/calculate_ece.py --M only_Ss --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1w_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1w_Ss --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_Ss --D CIFAR100_imb100

# sleep 60

python ECE/calculate_ece.py --M T1w_T2w_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1w_T2w_Ss --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_T2w_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_T2w_Ss --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1w_T2s_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1w_T2s_Ss --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_T2s_Sw --D CIFAR100_imb100
python ECE/calculate_ece.py --M T1s_T2s_Ss --D CIFAR100_imb100