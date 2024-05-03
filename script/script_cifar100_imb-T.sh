
python main.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr_steps 55 65 0 \
      --train_add_strong  \
      --add_name get_Ts

sleep 100

python main.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr_steps 55 65 0 \
      --train_add_weak  \
      --add_name get_Tw