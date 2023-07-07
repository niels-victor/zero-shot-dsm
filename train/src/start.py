from train.train import train
from train.setup import setup_cuda
from train.parse_args import args_to_config

setup_cuda()
config = args_to_config()
train(config)
