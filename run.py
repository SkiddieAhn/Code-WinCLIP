import os
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms as transforms
from dataset import MVTecDataset
from tabulate import tabulate
from fastprogress import progress_bar
from func_WinCLIP import CLIP_AD, harmonic_aggregation, compute_score, compute_sim, vis, prepare_text_future
from sklearn.metrics import roc_auc_score


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.dataset} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_data(model, img_size, dataset_name, dataset_dir, batch_size):
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    preprocess = model.preprocess
    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))

    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    else:
        print('There are no data!')
        return None
        
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_dataloader, obj_list


def win_clip(model, loader, avg_normal_text_features, avg_abnormal_text_features, device):
    results = {}
    results['cls_names'] = []
    results['imgs_paths'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    patch_size = 16

    length = len(loader)
    for items in progress_bar(loader, total=length): 
        images = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        results['cls_names'].extend(cls_name)
        img_path = items['img_path']
        results['imgs_paths'].extend(img_path)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].extend(items['anomaly'].detach().cpu())

        b, _, h, w = images.shape

        one_normal_features = avg_normal_text_features[cls_id]
        one_abnormal_features = avg_abnormal_text_features[cls_id]

        # feature extraction using windows
        large_scale_tokens, mid_scale_tokens, _, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)

        # multi-scale embedding map
        zscore = compute_score(class_tokens, torch.cat((one_normal_features, one_abnormal_features), dim = 1).permute(0, 2, 1))
        z0score = zscore[:,0,1]
        large_scale_simmarity = compute_sim(large_scale_tokens, torch.cat((one_normal_features, one_abnormal_features), dim = 1).permute(0, 2, 1))[:,:,1].to(device)
        mid_scale_simmarity = compute_sim(mid_scale_tokens, torch.cat((one_normal_features, one_abnormal_features), dim = 1).permute(0, 2, 1))[:,:,1].to(device)

        # multi-scale aggregation
        large_scale_score = harmonic_aggregation((b, h//patch_size, w//patch_size) ,large_scale_simmarity, large_scale)
        mid_scale_score  = harmonic_aggregation((b, h//patch_size, w//patch_size), mid_scale_simmarity, mid_scale)
        multiscale_score = mid_scale_score
        multiscale_score = torch.nan_to_num(3.0/(1.0/large_scale_score + 1.0/mid_scale_score + 1.0/z0score.unsqueeze(1).unsqueeze(1)), nan=0.0, posinf=0.0, neginf=0.0)
        
        # interpolate
        multiscale_score = multiscale_score.unsqueeze(1).to(device)
        multiscale_score = F.interpolate(multiscale_score, size=(h, w), mode='bilinear')
        multiscale_score = multiscale_score.squeeze()

        # save
        results['pr_sp'].extend(z0score.detach().cpu())
        results['anomaly_maps'].append(multiscale_score)
    
    results['imgs_masks'] = torch.cat(results['imgs_masks'])
    results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()
    return results


def calc_AUC(results, obj_list):
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []

    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []

        table.append(obj)
        for idxs in range(len(results['cls_names'])):
            if results['cls_names'][idxs] == obj:
                gt_px.append(results['imgs_masks'][idxs].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxs])
                gt_sp.append(results['gt_sp'][idxs])
                pr_sp.append(results['pr_sp'][idxs])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)

    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'pixel_AUC', 'image_AUC'], tablefmt="pipe")
    print(results)    


@torch.no_grad()
def test(cfg):
    # setup default environment
    img_size = cfg.image_size
    dataset_name = cfg.dataset
    dataset_dir = cfg.data_path
    batch_size = cfg.batch_size
    model_name = cfg.model
    load_results = cfg.load_results
    visualization = cfg.vis
    mode = 'attn' if cfg.attn_mode else 'heat'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.print_cfg() 

    if load_results:
        with open(f'results/{dataset_name}/{load_results}.pickle', 'rb') as f:
            results = pickle.load(f)
        
        if dataset_name == 'mvtec':
            obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                        'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']

    else:
        # load model
        model = CLIP_AD(img_size, model_name)
        model.to(device)
        print('=================================================')
        print('1. load model ok!')
        print('=================================================')

        # load data
        loader, obj_list = get_data(model, img_size, dataset_name, dataset_dir, batch_size)
        print('=================================================')
        print('2. preparing dataloader ok!')
        print('=================================================')

        # make text features 
        avg_normal_text_features, avg_abnormal_text_features = prepare_text_future(model, obj_list)
        print('=================================================')
        print('3. making text features ok!')
        print('=================================================')

        # run WinCLIP method
        results = win_clip(model, loader, avg_normal_text_features, avg_abnormal_text_features, device) 
        with open(f'results/{dataset_name}/{dataset_name}_results.pickle', 'wb') as f:
            pickle.dump(results, f)
        print('=================================================')
        print('4. WinCLIP AD results save ok!')
        print('=================================================')

    # calc AUC (image-level, pixel-level)
    calc_AUC(results, obj_list)
    print('=================================================')
    print('[Evaluate] calculate AUC ok!')
    print('=================================================')

    # visualization
    if visualization:
        vis(paths=results['imgs_paths'], anomaly_map=results['anomaly_maps'], img_size=img_size, save_path=f'results/{dataset_name}', cls_name=results['cls_names'], mode=mode)
        print('=================================================')
        print('[Evaluate] visualization ok!')
        print('=================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/sha/datasets/mvtec", help="path to test dataset")
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--load_results", type=str, default=None)
    parser.add_argument("--vis", type=str2bool, nargs='?', const=False)
    parser.add_argument("--attn_mode", type=str2bool, nargs='?', const=False)
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()

    if not os.path.exists(f'results'):
        os.mkdir(f'results')
    if not os.path.exists(f'results/{args.dataset}'):
        os.mkdir(f'results/{args.dataset}')

    print('---------------------------------')
    print('GPU:', torch.cuda.is_available())
    print('---------------------------------')

    setup_seed(args.seed)
    config_dict = vars(args)
    test(dict2class(config_dict))
