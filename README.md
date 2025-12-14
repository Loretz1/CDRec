# CDRec
## RUN
python -u main.py --model DisenCDR --dataset sport_clothing
nohup python -u main.py --model Base --dataset sport_clothing >run.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --model DisenCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors >run.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --model DisenCDR --dataset Amazon2014 --domains Sports_and_Outdoors Clothing_Shoes_and_Jewelry >run1.log 2>&1 &