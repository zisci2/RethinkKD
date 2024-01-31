
python main.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 60 \
      --lr_CosineAnnealing 35 50 \
      --train_add_strong  \
      --add_name get_Ts

sleep 100

python main.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 60 \
      --lr_CosineAnnealing 30 0 \
      --train_add_weak  \
      --add_name get_Tw