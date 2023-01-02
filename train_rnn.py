import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.seq2seq import SimpleRNN
from src.config import Config
from src.scaler import Wrapper
from src.utils import split_data, plot_learning_curve
from src.dataset import CustomDataset
from src.train import train, evaluate
from torch.utils.data import DataLoader

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training process for time series forecasting")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "lstm")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--num_epoch", type = int, default = 64)
    parser.add_argument("--seq_len", type = int, default = 128)
    parser.add_argument("--pred_len", type = int, default = 32)
    parser.add_argument("--stride", type = int, default = 4)
    parser.add_argument("--dist", type = int, default = 0)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = False)

    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW")
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    
    # scaler : MinMax, Robust, Standard, BatchNorm, LayerNorm, RevIN
    parser.add_argument("--use_scaler", type = bool, default = True)
    parser.add_argument("--scaler", type = str, default = "RevIN", choices=['MinMax','BatchNorm', 'LayerNorm', 'InstanceNorm','RevIN'])

    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = 4)
    
    # model setup - lstm
    parser.add_argument("--hidden_dim", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 2)
    parser.add_argument("--teacher_forcing_ratio", type = float, default = 0.5)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    # argument
    args = parsing()
    
    # Default configuration
    config = Config()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    # scaler
    if not args['use_scaler']:
        args_scaler = {}
        tag_scale = 'None'
        
    elif args['scaler'] == 'MinMax':
        args_scaler = {            
            "num_features" : len(config.src_cols),
            "eps" : 1e-6
        }
        tag_scale = 'MinMax'
    elif args['scaler'] == 'BatchNorm':
        args_scaler = {            
            "num_features" : len(config.src_cols),
            "eps" : 1e-6,
            "momentum" : 0.1,
            "affine" : True,
            "track_running_stats" : True
        }
        tag_scale = 'BN'
    elif args['scaler'] == 'InstanceNorm':
        args_scaler = {            
            "num_features" : len(config.src_cols),
            "eps" : 1e-6,
            "momentum" : 0.1,
            "affine" : True,
            "track_running_stats" : True
        }
        tag_scale = 'IN'
    elif args['scaler'] == 'LayerNorm':
        args_scaler = {            
            "num_features" : len(config.src_cols),
            "eps" : 1e-6,
            "gamma":True, 
            "beta":True
        }
        tag_scale = 'LN'
    elif args['scaler'] == 'RevIN':
        args_scaler = {
            "num_features" : len(config.src_cols),
            "eps" : 1e-6,
            "affine" : True
        }
        tag_scale = 'RevIN'
    else:     
        args_scaler = {
            "num_features" : len(config.src_cols),
            "eps" : 1e-6,
            "affine" : True
        }
        tag_scale = 'RevIN'
        
    # directory setting
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    tag = "{}_seq_{}_pred_{}_dist_{}_scaler_{}".format(args["tag"], args["seq_len"], args['pred_len'], args["dist"], tag_scale)
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
    exp_dir = os.path.join(save_dir, "tensorboard_{}".format(tag))
    
    # dataset setup
    data = pd.read_csv(config.DATA_PATH['ettm1'])
    ts_train, ts_valid, ts_test = split_data(data, 0.6, 0.2)
    
    train_data = CustomDataset(ts_train, config.src_cols, config.tar_cols, args['seq_len'], args['pred_len'], stride = args['stride'])
    valid_data = CustomDataset(ts_valid, config.src_cols, config.tar_cols, args['seq_len'], args['pred_len'], stride = args['stride'])
    test_data = CustomDataset(ts_test, config.src_cols, config.tar_cols, args['seq_len'], args['pred_len'], stride = args['stride'])

    # dataset info
    print("train data : {}, # of features : {}".format(train_data.__len__(), len(config.src_cols)))
    print("valid data : {}, # of features : {}".format(valid_data.__len__(), len(config.src_cols)))
    print("test data : {}, # of features : {}".format(test_data.__len__(), len(config.src_cols)))
    
    # define model
    network = SimpleRNN(
        input_dim = len(config.src_cols),
        hidden_dim = args['hidden_dim'],
        output_dim = len(config.tar_cols),
        n_layers = args['n_layers'],
        target_len = args['pred_len'],
        teacher_forcing_ratio=args['teacher_forcing_ratio']
    )
    
    print("\n################# model summary #################\n")
    network.summary()
    
    model = Wrapper(
        network,
        tag_scale,
        **args_scaler
    )
    
    model.to(device)

    # optimizer
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
        
    # scheduler
    if args["use_scheduler"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
    else:
        scheduler = None
        
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], shuffle = True, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], shuffle = True,  num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], shuffle = True,  num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    # loss
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # training process
    print("\n################# training process #################\n")
    train_loss, valid_loss = train(
        train_loader, 
        valid_loader,
        model,
        'lstm',
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch = args['num_epoch'],
        verbose = args['verbose'],
        save_dir = "./weights",
        tag = tag,
        max_norm_grad = 1.0
    )
    
    # plot the learning curve
    save_learning_curve = os.path.join(save_dir, "{}_lr_curve.png".format(tag))
    plot_learning_curve(train_loss, valid_loss, figsize = (8,6), save_dir = save_learning_curve)
    
    # evaluation process
    print("\n################# evaluation process #################\n")
    model.load_state_dict(torch.load(os.path.join("./weights", "{}_best.pt".format(tag))))
    test_loss = evaluate(
        test_loader,
        model,
        'lstm',
        optimizer,
        loss_fn,
        device, 
    )