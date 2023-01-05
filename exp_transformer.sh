# experiment : performance comparsion with different scaler
# Normalization : LayerNorm, BatchNorm, MinMax, InstanceNorm
# Model - Transformer

python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'Normal'
python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'MinMax'
python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'BatchNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'LayerNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'InstanceNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth1' --scaler 'RevIN'

python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'Normal'
python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'MinMax'
python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'BatchNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'LayerNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'InstanceNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'etth2' --scaler 'RevIN'

python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'Normal'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'MinMax'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'BatchNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'LayerNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'InstanceNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm1' --scaler 'RevIN'

python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'Normal'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'MinMax'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'BatchNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'LayerNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'InstanceNorm'
python3 train_transformer.py --gpu_num 3 --dataset 'ettm2' --scaler 'RevIN'