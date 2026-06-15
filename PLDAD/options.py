import argparse
import os
import torch

class Options:

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--seed', type=int, default=12345, help='random seed')
        self.parser.add_argument('--dataset', default='SMD', help='dataset')
        self.parser.add_argument('--filename', default='machine-1-1.txt', help='dataset name')
        self.parser.add_argument('--train_batchsize', type=int, default=128, help='batch size of train data')
        self.parser.add_argument('--val_batchsize', type=int, default=128, help='batch size of validation data')
        self.parser.add_argument('--test_batchsize', type=int, default=128, help='batch size of test data')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        self.parser.add_argument('--prefetch_factor', type=int, default=2,
                     help='DataLoader prefetch factor (effective when workers > 0)')
        self.parser.add_argument('--step', type=int, default=1, help='sequence step')  
        self.parser.add_argument('--window_size', type=int, default=4, help='sequence length')
        self.parser.add_argument('--dim', type=int, default=38, help='dimensions of data')

        self.parser.add_argument('--project_channels', type=int, default=32, help='channels of project')
        self.parser.add_argument('--drop_out', type=float, default=0.2, help='dropout')

        self.parser.add_argument('--train_stage', type=str, default='jft',
                                 help='training stage: mfae | lpl | jft')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--deterministic', action='store_true',
                     help='Enable deterministic CUDA mode (slower, more reproducible)')
        self.parser.add_argument('--allow_tf32', type=int, default=1,
                     help='Enable TF32 on Ampere+ GPU for faster matmul/cudnn (1/0)')
        self.parser.add_argument('--matmul_precision', type=str, default='high',
                     choices=['highest', 'high', 'medium'],
                     help='torch float32 matmul precision hint')
        self.parser.add_argument('--model', type=str, default='SCAD', help='detection model')
        self.parser.add_argument('--outf', default='./output', help='output folder')

        self.parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train for')

        self.parser.add_argument('--lr_mfae', type=float, default=1e-3,
                                 help='Stage 1 (MFAE) learning rate')
        self.parser.add_argument('--lr_lpl', type=float, default=1e-4,
                                 help='Stage 2 (LPL) learning rate')
        self.parser.add_argument('--lr_jft', type=float, default=1e-5,
                                 help='Stage 3 (JFT) learning rate')

        self.parser.add_argument('--patience', type=int, default=5, help='early stopping')

        self.parser.add_argument('--run_number', type=int, default=1,
                                 help='运行次数，用于区分不同运行的结果文件。')

        self.parser.add_argument('--diff_beta_schedule', type=str, default='linear',
                                 help='扩散模型的beta调度方式: linear | cosine')
        self.parser.add_argument('--diff_timesteps', type=int, default=100,
                                 help='扩散模型的时间步数')
        self.parser.add_argument('--diff_beta_start', type=float, default=1e-5,
                                 help='扩散模型beta起始值')
        self.parser.add_argument('--diff_beta_end', type=float, default=1e-2,
                                 help='扩散模型beta结束值')
        self.parser.add_argument('--diff_noise_type', type=str, default='default',
                                 help='扩散模型噪声类型')
        self.parser.add_argument('--n_copies_to_test', type=int, default=1,
                                 help='测试时的复制数量')
        self.parser.add_argument('--test_batch_size', type=int, default=64,
                                 help='测试批次大小')
        self.parser.add_argument('--diff_hidden_dim', type=int, default=32,
                                 help='扩散模型隐藏维度')
        self.parser.add_argument('--diff_n_heads', type=int, default=2,
                                 help='扩散模型注意力头数')
        self.parser.add_argument('--diff_attn_dropout', type=float, default=0.1,
                                 help='扩散模型注意力dropout率')
        self.parser.add_argument('--diff_mlp_ratio', type=float, default=4.0,
                                 help='扩散模型MLP比例')
        self.parser.add_argument('--diff_n_depth', type=int, default=2,
                                 help='扩散模型深度')
        self.parser.add_argument('--diff_n_emb', type=int, default=2,
                                 help='扩散模型嵌入维度')
        self.parser.add_argument('--mask_ratio', type=float, default=0.5,
                                 help='掩码率')
        self.parser.add_argument('--jitter_scale', type=float, default=0.5,
                                 help='抖动幅度(0.5 = +/- 25%)')
        self.parser.add_argument('--finetune_lr_scale', type=float, default=0.1, help='Stage 3 LR scale')

        self.parser.add_argument('--diffusion_loss_type', type=str, default='x0',
                                 choices=['x0', 'noise'],
                                 help='扩散损失类型: x0(直接MSE,推荐) | noise(DDPM原始)')
        self.parser.add_argument('--jft_diff_weight', type=float, default=1.0,
                     help='Stage 3 diffusion loss weight: total_loss = w*loss_diff + loss_rec')

        self.parser.add_argument('--suffix', type=str, default='default',
                                 help='自定义文件名后缀，用于区分不同实验配置')

        self.opt = None

    def parse(self):

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if self.opt.device == 'gpu':
            if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])
            else:
                print('Warning: CUDA unavailable or gpu_ids is empty, fallback to CPU mode')
                self.opt.device = 'cpu'

        args = vars(self.opt)  

        self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
