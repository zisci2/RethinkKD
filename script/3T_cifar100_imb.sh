# T1w_T2w_T3w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_weak --T3_add_weak --S_add_weak \
   --add_name T1w_T2w_T3w_Sw \

sleep 300

# T1w_T2w_T3w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_weak --T3_add_weak --S_add_strong \
   --add_name T1w_T2w_T3w_Ss \


sleep 300


# T1s_T2w_T3w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --T3_add_weak --S_add_weak \
   --add_name T1s_T2w_T3w_Sw \


sleep 300

# T1w_T2s_T3w_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --T3_add_weak --S_add_weak \
   --add_name T1w_T2s_T3w_Sw \


# T1w_T2w_T3s_Sw 
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_weak --T3_add_strong --S_add_weak \
   --add_name T1w_T2w_T3s_Sw \


sleep 300

# T1s_T2w_T3w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --T3_add_weak --S_add_strong \
   --add_name T1s_T2w_T3w_Ss \


sleep 300

# T1w_T2s_T3w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --T3_add_weak --S_add_strong \
   --add_name T1w_T2s_T3w_Ss \


sleep 300

# T1w_T2w_T3s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_weak --T3_add_strong --S_add_strong \
   --add_name T1w_T2w_T3s_Ss \


sleep 300

# T1s_T2s_T3w_Sw 
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --T3_add_weak --S_add_weak \
   --add_name T1s_T2s_T3w_Sw \


sleep 300

# T1w_T2s_T3s_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --T3_add_strong --S_add_weak \
   --add_name T1w_T2s_T3s_Sw \


sleep 300

# T1s_T2w_T3s_Sw 
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --T3_add_strong --S_add_weak \
   --add_name T1s_T2w_T3s_Sw \

sleep 300

# T1s_T2s_T3w_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --T3_add_weak --S_add_strong \
   --add_name T1s_T2s_T3w_Ss \


sleep 300

# T1w_T2s_T3s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_weak --T2_add_strong --T3_add_strong --S_add_strong \
   --add_name T1w_T2s_T3s_Ss \

sleep 300

# T1s_T2w_T3s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_weak --T3_add_strong --S_add_strong \
   --add_name T1s_T2w_T3s_Ss \

sleep 300



# T1s_T2s_T3s_Sw
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --T3_add_strong --S_add_weak \
   --add_name T1s_T2s_T3s_Sw \


sleep 300

# T1s_T2s_T3s_Ss
python main.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 175 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T2_add_strong --T3_add_strong --S_add_strong \
   --add_name T1s_T2s_T3s_Ss \


sleep 300

# #####################



















