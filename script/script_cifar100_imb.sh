####### 1T

#T1w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
    --epoch 175 --temp 3.0 --alpha 0.6 \
    --lr_steps 160 165 170 \
    --T1_add_weak --S_add_weak \
    --add_name T1w_Sw \

#T1w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
    --epoch 175 --temp 3.0 --alpha 0.6 \
    --lr_steps 160 165 170 \
    --T1_add_weak --S_add_strong \
    --add_name T1w_Ss \

#T1s_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
    --epoch 175 --temp 3.0 --alpha 0.6 \
    --lr_steps 160 165 170 \
    --T1_add_strong --S_add_weak \
    --add_name T1s_Sw \

#T1s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
    --epoch 175 --temp 3.0 --alpha 0.6 \
    --lr_steps 160 165 170 \
    --T1_add_strong --S_add_strong \
    --add_name T1s_Ss \

############## 2T

# T1w_T2w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_weak --S_add_weak \
   --add_name T1w_T2w_Sw \
  #  --next_continue epoch_175/CIFAR100_distil_175_T1w_T2w_Sw

sleep 60

# # T1w_T2w_Ss
 python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
    --epoch 175 --temp 3.0 --alpha 0.6 \
    --lr_steps 160 165 170 \
    --T1_add_weak --T2_add_weak --S_add_strong \
    --add_name T1w_T2w_Ss \
#    --next_continue epoch_175/CIFAR100_distil_175_T1w_T2w_Ss

sleep 60

# T1s_T2w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --S_add_weak \
   --add_name T1s_T2w_Sw \
#    --next_continue epoch_175/CIFAR100_distil_175_T1s_T2w_Sw

sleep 60

# T1s_T2w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --S_add_strong \
   --add_name T1s_T2w_Ss \
#    --next_continue epoch_175/CIFAR100_distil_175_T1s_T2w_Ss

sleep 60


# T1w_T2s_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --S_add_weak \
   --add_name T1w_T2s_Sw \
#    --next_continue epoch_175/CIFAR100_distil_175_T1w_T2s_Sw

sleep 60


# T1w_T2s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --S_add_strong \
   --add_name T1w_T2s_Ss \
#    --next_continue epoch_175/CIFAR100_distil_175_T1w_T2s_Ss

sleep 60


# T1s_T2s_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --S_add_weak \
   --add_name T1s_T2s_Sw \
#    --next_continue epoch_175/CIFAR100_distil_175_T1s_T2s_Sw

sleep 60

# T1s_T2s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --S_add_strong \
   --add_name T1s_T2s_Ss \
#    --next_continue epoch_175/CIFAR100_distil_175_T1s_T2s_Ss

sleep 60