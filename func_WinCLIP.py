import torch
import torch.nn as nn 
import os
import cv2
import numpy as np 
from src import open_clip
from open_clip import tokenizer
import matplotlib.pyplot as plt


class prompt_order():
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list =[
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of the {}.",
        "a dark photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection."
        ]

    def prompt(self, class_name):
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template
    


class patch_scale():
    def __init__(self, image_size):
        self.h, self.w = image_size
 
    def make_mask(self, patch_size = 16, kernel_size = 16, stride_size = 16): 
        self.patch_size = patch_size
        self.patch_num_h = self.h//self.patch_size
        self.patch_num_w = self.w//self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size//patch_size
        self.stride_size = stride_size//patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype = torch.float32).reshape((1,1,self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy
    

class CLIP_AD(nn.Module):
    def __init__(self, image_size, model_name = 'ViT-B-16-plus-240'):
        super(CLIP_AD, self).__init__()
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.mask = patch_scale((image_size, image_size))

    def multiscale(self):
        pass
    
    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image, patch_size, mask=True):
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale,mid_scale], proj = False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale


def harmonic_aggregation(score_size, similarity, mask):
    b, h, w = score_size
    similarity = similarity.double()
    score = torch.zeros((b, h*w)).to(similarity).double()
    mask = mask.T
    for idx in range(h*w):
        patch_idx = [bool(torch.isin(idx+1, mask_patch)) for mask_patch in mask]
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / similarity[:, patch_idx], dim = -1)
        score[:, idx] =sum_num /harmonic_sum
    score = score.reshape(b, h, w)
    return score


def compute_score(image_features, text_features):
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)
    return text_probs


def compute_sim(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    similarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return similarity


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
    

def apply_ad_scoremap(image, scoremap, alpha=0.5, mode='seg'):
    if mode == 'heat':
        np_image = np.asarray(image, dtype=float)
        scoremap = np.repeat(scoremap[:,:,np.newaxis], 3, axis=2)
        applied = (np_image * scoremap).astype(np.uint8)
        return applied
    else:
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def vis(paths, anomaly_map, img_size, save_path, cls_name, mode):
    for idx, path in enumerate(paths):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]

        # load image and transform to RGB 
        vis_original = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        
        # apply anomaly score map
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(image=vis_original, scoremap=mask, mode=mode)

        # combine image
        combined = cv2.hconcat([vis_original, vis])

        # save image
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)  # BGR
        save_img_path = os.path.join(save_path, 'imgs', f'{mode}', cls_name[idx], cls)
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        cv2.imwrite(os.path.join(save_img_path, filename), combined)


def prepare_text_future(model, obj_list):
    Memory_avg_normal_text_features = []
    Memory_avg_abnormal_text_features = []
    text_generator = prompt_order()

    for i in obj_list:
        normal_description, abnormal_description = text_generator.prompt(i)

        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)
        Memory_avg_normal_text_features.append(avg_normal_text_features)
        Memory_avg_abnormal_text_features.append(avg_abnormal_text_features)

    Memory_avg_normal_text_features = torch.stack(Memory_avg_normal_text_features)       
    Memory_avg_abnormal_text_features = torch.stack(Memory_avg_abnormal_text_features)  
    return Memory_avg_normal_text_features, Memory_avg_abnormal_text_features
