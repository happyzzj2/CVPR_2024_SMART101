from model import clip_Model,bridge_atten_Model,deepspeed_trainModel
import argparse
import pdb
import os 
import torch
import numpy as np
# import deepspeed


def set_args():
    parser = argparse.ArgumentParser(description="SMART dataset")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size (16)")
    parser.add_argument("--num_epochs", default=4, type=int, help="epoch")
    parser.add_argument("--patience", default=3, type=int, help="early stop epoch") 
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--data_root",type=str,default="./SMART101-release-v1/SMART101-Data/",help="location of the csv files, and location of the images, relative location is provided in the csv file.")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--seed", type=int, default=1, help="seed to use")
    parser.add_argument("--save_path", type=str, default="./result/", help="location to save intermediate files.")
    parser.add_argument("--date", type=str, default="2024_4_9", help="date of saving model.")
    parser.add_argument("--hyperp_flag", type=str, default="a", help="hyper parameter flag")
    parser.add_argument("--method", type=str, default="VL_encoder", help="method") 
    parser.add_argument("--cnt_layer", type=int, default=2, help="cnt encoder layer")  
    parser.add_argument('--local_rank', type=int, default=-1,help='local rank passed from distributed launcher')
    # parser.add_argument('--deepspeed', '-d', action='store_true',help='Enable DeepSpeed (requires a config file)')
    # parser.add_argument('--deepspeed_config', '-ds_conf', type=str, default='./ds_config.json',help='DeepSpeed config file path')
    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def reset_state(args):
    """
    重置初始化参数
    :param args:参数
    """
    manualSeed = np.random.randint(10000) if args.seed == -1 else args.seed
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (args.seed))

if __name__ == "__main__":
    # 参数
    args = set_args()
    
    reset_state(args) # 生成随机数种子
    # model = clip_Model(args)
    model = bridge_atten_Model(args)
    # model = deepspeed_trainModel(args)
    # 训练
    model.train()
