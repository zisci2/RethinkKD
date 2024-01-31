
python main.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --train_add_strong  \
      --add_name get_Ts

sleep 100

python main.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --train_add_weak  \
      --add_name get_Tw