# 计算T与S的IoU

# T1w_T2w_Sw 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1w_T2w_Sw \
    --dataset ImageNet_LT \
    --add_name T1w_T2w_Sw

# T1w_T2w_Ss 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1w_T2w_Ss \
    --model_name T1w_T2w_Ss \
    --dataset ImageNet_LT \
    --add_name T1w_T2w_Ss

# T1s_T2w_Sw 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1s_T2w_Sw \
    --dataset ImageNet_LT \
    --add_name T1s_T2w_Sw

# T1s_T2w_Ss 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1s_T2w_Ss \
    --dataset ImageNet_LT \
    --add_name T1s_T2w_Ss

# T1w_T2s_Sw 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1w_T2s_Sw \
    --dataset ImageNet_LT \
    --add_name T1w_T2s_Sw

# T1w_T2s_Ss 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1w_T2s_Ss \
    --dataset ImageNet_LT \
    --add_name T1w_T2s_Ss

# T1s_T2s_Sw 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1s_T2s_Sw \
    --dataset ImageNet_LT \
    --add_name T1s_T2s_Sw

# T1s_T2s_Ss 基线
python calculate-IoU.py \
    --mode T1_T2 \
    --model_name T1s_T2s_Ss \
    --dataset ImageNet_LT \
    --add_name T1s_T2s_Ss

#####################################################################
