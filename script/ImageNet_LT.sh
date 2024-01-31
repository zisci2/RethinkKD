# T1w_T2w_Sw
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_weak --T2_add_weak --S_add_weak \
   --add_name T1w_T2w_Sw \
  #  --next_continue 

sleep 60

# # T1w_T2w_Ss
 python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
    --epoch 165 --temp 3.0 --alpha 0.6 \
    --lr_steps 150 155 160 \
    --T1_add_weak --T2_add_weak --S_add_strong \
    --add_name T1w_T2w_Ss \
#    --next_continue 

sleep 60

# T1s_T2w_Sw
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T2_add_weak --S_add_weak \
   --add_name T1s_T2w_Sw \
#    --next_continue 

sleep 60

# T1s_T2w_Ss
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T2_add_weak --S_add_strong \
   --add_name T1s_T2w_Ss \
#    --next_continue 

sleep 60


# T1w_T2s_Sw
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_weak --T2_add_strong --S_add_weak \
   --add_name T1w_T2s_Sw \
#    --next_continue 

sleep 60


# T1w_T2s_Ss
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_weak --T2_add_strong --S_add_strong \
   --add_name T1w_T2s_Ss \
#    --next_continue

sleep 60


# T1s_T2s_Sw
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T2_add_strong --S_add_weak \
   --add_name T1s_T2s_Sw \
#    --next_continue 

sleep 60

# T1s_T2s_Ss
python main.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T2_add_strong --S_add_strong \
   --add_name T1s_T2s_Ss \
#    --next_continue 

sleep 60