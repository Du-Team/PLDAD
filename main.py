import time

import torch
import numpy as np
from sklearn.metrics import average_precision_score

from PLDAD.data_utils import *
from PLDAD.eval_methods import bf_search
from PLDAD.options import Options
from PLDAD.model import *
from torch.utils.data import DataLoader

import os

opt = Options().parse()

def select_runtime_device(opt_args):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if str(getattr(opt_args, 'device', 'gpu')).lower() == 'cpu':
        return torch.device("cpu")

    gpu_ids = getattr(opt_args, 'gpu_ids', [0])
    if isinstance(gpu_ids, str):
        gpu_ids = [int(x) for x in gpu_ids.split(',') if int(x) >= 0]

    gpu_index = int(gpu_ids[0]) if len(gpu_ids) > 0 else 0
    available_gpu_count = torch.cuda.device_count()
    if gpu_index >= available_gpu_count:
        raise ValueError(f"Requested GPU index {gpu_index} but only {available_gpu_count} CUDA device(s) are available")

    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")

device = select_runtime_device(opt)
opt.device = device
if device.type == 'cuda':
    print(f"Using CUDA device {device.index}: {torch.cuda.get_device_name(device.index)}")
else:
    print("Using CPU device")

def configure_torch_runtime():
    use_cuda = isinstance(opt.device, torch.device) and opt.device.type == 'cuda'
    deterministic = bool(getattr(opt, 'deterministic', False))

    if use_cuda:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

        allow_tf32 = bool(getattr(opt, 'allow_tf32', 1))
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = allow_tf32

        if hasattr(torch, 'set_float32_matmul_precision'):
            try:
                torch.set_float32_matmul_precision(getattr(opt, 'matmul_precision', 'high'))
            except Exception:
                pass

def set_seed(seed_value):
    if seed_value == -1:
        return

    import random

    random.seed(seed_value)

    torch.manual_seed(seed_value)
    if isinstance(opt.device, torch.device) and opt.device.type == 'cuda':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

configure_torch_runtime()

set_seed(int(opt.seed))

def anomaly_detection(data_type):
    if not os.path.exists('./results_label'):
        os.makedirs('./results_label')

    if not os.path.exists('./train_model'):
        os.makedirs('./train_model')

    if not os.path.exists('./results_metric'):
        os.makedirs('./results_metric')

    file_name_1 = f"{data_type}_window_{opt.window_size}_{opt.suffix}_run_{opt.run_number}"

    opt.dataset = data_type

    if data_type == 'SMAP':
        opt.dim = 25

    elif data_type == 'MSL':
        opt.dim = 55

    elif data_type == 'SMD':
        opt.dim = 38

    elif data_type == 'SWAT':
        opt.dim = 51
    elif data_type == 'PSM':
        opt.dim = 25
        opt.step = 1
    elif data_type == 'Genesis':
        opt.dim = 18
        opt.step = 1
    elif data_type == 'ASD':
        opt.dim = 19
        opt.step = 1
    else:
        print("There is no this dataset!!!!")

    path_train = os.path.join(os.getcwd(), "datasets", "train", data_type)
    files = sorted(os.listdir(path_train))
    processed_files = []

    stage_prefix = {'mfae': '1', 'lpl': '2', 'jft': '3'}
    stage_tag = getattr(opt, 'train_stage', 'jft')
    prefix = stage_prefix[stage_tag]
    result_path = f"./results_metric/{stage_tag}_results_{prefix}_{file_name_1}.txt"
    f = open(result_path, 'w')
    f.write('channel' + '\t' + 'dataset' + '\t' + 'f1' + '\t' + 'pre' + '\t' + 'rec' + '\t' +
            'tp' + '\t' + 'tn' + '\t' + 'fp' + '\t' + 'fn' + '\t' + 'train_time' + '\t' + 'epoch_time' +
            '\t' + 'test_time' + '\t' + 'threshold' + '\t' + 'auc' + '\t' + 'ap' + '\n')

    for file in files:
            if file in processed_files:
                print(f"skip processed file: {file}")
                continue
            opt.filename = file
            set_seed(int(opt.seed))
            data_name = data_type + '/' + str(file)
            print('file:', data_name)

            f_name = data_name.split('/')[1].split('.')[0]

            file_name_2 = f"{opt.dataset}_{f_name}_window_{opt.window_size}_{opt.suffix}_{opt.run_number}"

            def build_stage_model_path(stage_name, run_number):
                prefix = stage_prefix[stage_name]
                return f"./train_model/{stage_name}_model_{prefix}_{file_name_2}.pth"

            def build_stage_label_path(stage_name, run_number):
                prefix = stage_prefix[stage_name]
                return f"./results_label/{stage_name}_label_{prefix}_{file_name_2}.txt"

            samples_train_data, samples_val_data = read_train_data(opt.window_size, file=data_name,
                                                                   step=opt.step)
            print('train samples', samples_train_data.shape)
            print('valid samples', samples_val_data.shape)

            max_workers = os.cpu_count() or 1
            num_workers = max(0, min(int(opt.workers), max_workers))
            loader_kwargs = {
                'num_workers': num_workers,
                'pin_memory': torch.cuda.is_available(),
            }
            if num_workers > 0:
                loader_kwargs['persistent_workers'] = True
                loader_kwargs['prefetch_factor'] = max(2, int(getattr(opt, 'prefetch_factor', 2)))

            train_data = DataLoader(
                dataset=samples_train_data,
                batch_size=opt.train_batchsize,
                shuffle=True,
                **loader_kwargs,
            )
            val_data = DataLoader(
                dataset=samples_val_data,
                batch_size=opt.val_batchsize,
                shuffle=True,
                **loader_kwargs,
            )
            samples_test_data, test_label = read_test_data(opt.window_size, file=data_name)
            print('test samples', samples_test_data.shape)

            test_data = DataLoader(
                dataset=samples_test_data,
                batch_size=opt.test_batchsize,
                **loader_kwargs,
            )
            model = SCAD(opt)
            model.train_stage = getattr(opt, 'train_stage', 'jft')
            model = to_device(model, device)

            total_train_time = 0.0
            total_epoch_time = time.time()
            total_test_time = 0.0
            epochs = 0
            last_t = []
            last_th = 0.0
            last_label = []
            last_point_label = []
            last_f1, step = 0.0, 0
            last_auc = 0.0
            last_ap = 0.0

            model_path = build_stage_model_path(stage_tag, opt.run_number)
            label_path = build_stage_label_path(stage_tag, opt.run_number)

            if stage_tag == 'mfae':
                train_time, history, epoch_time = training(opt, model, train_data, val_data, model_path)
                model.load_state_dict(torch.load(model_path, map_location=device))

            elif stage_tag == 'lpl':
                mfae_model_path = build_stage_model_path('mfae', opt.run_number)
                if os.path.exists(mfae_model_path):
                    print(f"Loading MFAE weights from {mfae_model_path}")
                    model.load_state_dict(torch.load(mfae_model_path, map_location=device), strict=False)
                    print("Freezing encoder and decoder, only training diffusion model...")
                    encoder_modules = [
                        model.ver_module, model.hor_module, model.ver_atte, model.hor_atte,
                        model.fusion_attention, model.fusion_linear, model.projection_head,
                        model.continuous_decoder
                    ]
                    for module in encoder_modules:
                        for param in module.parameters():
                            param.requires_grad = False

                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
                else:
                    raise FileNotFoundError(
                        f"Stage 2 (LPL) requires Stage 1 (MFAE) weights!\n"
                        f"Expected path: {mfae_model_path}\n"
                        f"Please run Stage 1 first with: --train_stage mfae --run_number {opt.run_number}"
                    )
                train_time, history, epoch_time = training(opt, model, train_data, val_data, model_path)
                model.load_state_dict(torch.load(model_path, map_location=device))

            elif stage_tag == 'jft':
                lpl_model_path = build_stage_model_path('lpl', opt.run_number)

                if os.path.exists(lpl_model_path):
                    print(f"Loading LPL weights from {lpl_model_path}")
                    model.load_state_dict(torch.load(lpl_model_path, map_location=device), strict=False)
                    print("Loaded complete LPL stage weights (encoder + decoder + diffusion)")

                    print("Unfreezing all parameters for joint fine-tuning...")
                    for param in model.parameters():
                        param.requires_grad = True

                    print("Pre-trained weights loaded. Starting Fine-tuning with reduced LR...")
                    original_lr = opt.lr_jft
                    finetune_lr = original_lr * opt.finetune_lr_scale
                    print(f"Fine-tuning LR: {finetune_lr} (original: {original_lr})")

                    opt.lr_jft = finetune_lr
                    train_time, history, epoch_time = training(opt, model, train_data, val_data, model_path)
                    opt.lr_jft = original_lr
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    raise FileNotFoundError(
                        f"Stage 3 (JFT) requires Stage 2 (LPL) weights!\n"
                        f"Expected path: {lpl_model_path}\n"
                        f"Please run Stage 2 first with: --train_stage lpl --run_number {opt.run_number}"
                    )

            model.eval()
            results, test_time = testing(model, test_data, opt=opt)
            windows_labels = []
            for i in range(len(test_label) - opt.window_size):
                windows_labels.append(list(np.int_(test_label[i:i + opt.window_size])))

            y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])
            y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

            if len(y_test) >= len(y_pred):
                y_test = y_test[:len(y_pred)]
            else:
                y_pred = y_pred[:len(y_test)]

            best_metrics, best_threshold, best_predict, best_point_label = bf_search(y_pred, y_test)
            print(f"BF Search F1={best_metrics[0]:.4f} (threshold={best_threshold:.4f})")

            auc = ROC(y_test, y_pred)[1]
            ap = average_precision_score(y_test, y_pred)

            try:
                with open(label_path, 'w') as label_name:
                    for i in range(len(best_predict)):
                        label_name.write(str(best_predict[i]) + '\n')
                print(f"Successfully wrote {len(best_predict)} predictions to {label_path}")
            except Exception as e:
                print(f"Warning: Failed to write label file {label_path}: {e}")

            print(str(data_type) + '\t' + str(file) + '\tf1=' + str(best_metrics[0]) + '\tpre=' + str(best_metrics[1]) +
                  '\trec=' + str(best_metrics[2]) + '\ttp=' + str(best_metrics[3]) + '\ttn=' + str(
                best_metrics[4]) + '\tfp=' + str(best_metrics[5]) +
                  '\tfn=' + str(best_metrics[6]))

            f.write(
                str(data_type) + '\t' + str(file) + '\t' + str(best_metrics[0]) + '\t' + str(best_metrics[1]) + '\t' +
                str(best_metrics[2]) + '\t' + str(best_metrics[3]) + '\t' + str(best_metrics[4]) + '\t' + str(
                    best_metrics[5]) + '\t' + str(best_metrics[6]) +
                '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) + '\t' + str(
                    best_threshold) + '\t' + str(auc) + '\t' + str(ap) + '\n')
    f.close()
    print('finished')

if __name__ == '__main__':
    anomaly_detection(opt.dataset)
