from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import base64
from io import BytesIO

class Helper():
    def __init__(self):
        pass
    
    def label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def cross_entropy_for_onehot(self, pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

    def weights_init(self, m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    
    def check_similarity(self, ground_truth, reconstructed):
        ground_truth = np.array(ground_truth)
        reconstructed = np.array(reconstructed)
        
        ground_truth = (ground_truth * 255).astype(np.uint8)
        reconstructed = (reconstructed * 255).astype(np.uint8)
        
        mse = mean_squared_error(ground_truth, reconstructed)
        psnr = peak_signal_noise_ratio(ground_truth, reconstructed)
        ssim, _ = structural_similarity(ground_truth, reconstructed, full=True, win_size=3)
        
        return mse, psnr, ssim
    
    def encode_image(self, image_data):
        buffer = BytesIO()
        image_data.save(buffer, format='PNG')
        return base64.b85encode(buffer.getvalue()).decode()
    
    def decode_image(self, b85_string):
        return base64.b85decode(b85_string)
    
class LocalSession():
    def __init__(self):
        self.results=[]
    
    def add(self, item):
        self.results.append(item)

    def get_results(self):
        return self.results
    
    def remove(self, item):
        self.results.remove(item)
        
    def clear(self):
        self.results = []
