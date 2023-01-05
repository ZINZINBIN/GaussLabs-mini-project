python3 train_rnn.py --gpu_num 0 --dataset 'etth1' --scaler 'Normal'
python3 train_rnn.py --gpu_num 0 --dataset 'etth2' --scaler 'Normal'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm1' --scaler 'Normal'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm2' --scaler 'Normal'

python3 train_rnn.py --gpu_num 0 --dataset 'etth1' --scaler 'MinMax'
python3 train_rnn.py --gpu_num 0 --dataset 'etth2' --scaler 'MinMax'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm1' --scaler 'MinMax'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm2' --scaler 'MinMax'

python3 train_rnn.py --gpu_num 0 --dataset 'etth1' --scaler 'BatchNorm'
python3 train_rnn.py --gpu_num 0 --dataset 'etth2' --scaler 'BatchNorm'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm1' --scaler 'BatchNorm'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm2' --scaler 'BatchNorm'

python3 train_rnn.py --gpu_num 0 --dataset 'etth1' --scaler 'RevIN'
python3 train_rnn.py --gpu_num 0 --dataset 'etth2' --scaler 'RevIN'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm1' --scaler 'RevIN'
python3 train_rnn.py --gpu_num 0 --dataset 'ettm2' --scaler 'RevIN'