import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from src.utils import Helper

class LeNet(nn.Module):
    def __init__(self, activation_function='sigmoid'):
        super(LeNet, self).__init__()
        if activation_function == 'relu':
            act = nn.ReLU
        elif activation_function == 'sigmoid':
            act = nn.Sigmoid
        elif activation_function == 'tanh':
            act = nn.Tanh
        else:
            act = nn.Sigmoid
            
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
class ImageProcessing():
    
    def __init__(self):
        # Initialize CIFAR and transforms
        self.dst = datasets.CIFAR100("~/.torch", download=True)
        self.tp = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        self.tt = transforms.ToPILImage()
        if torch.cuda.is_available():
            self.device = "cuda"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process_single_image(self, idx, activation_function):
        helper = Helper()
        net = LeNet(activation_function).to(self.device)
        net.apply(helper.weights_init)
        criterion = helper.cross_entropy_for_onehot
        
        # Process the image
        gt_data = self.tp(self.dst[idx][0]).to(self.device)
        gt_data = gt_data.view(1, *gt_data.size())
        gt_label = torch.Tensor([self.dst[idx][1]]).long().to(self.device)
        gt_label = gt_label.view(1, )
        gt_onehot_label = helper.label_to_onehot(gt_label)
        
        # Compute original gradient
        out = net(gt_data)
        y = criterion(out, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        
        # Generate dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(self.device).requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to(self.device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        
        # Training loop
        for _ in range(300):
            def closure():
                optimizer.zero_grad()
                pred = net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)
        
        # Get results
        reconstructed_data = self.tt(dummy_data[0].cpu())
        mse, psnr, ssim = helper.check_similarity(self.tt(gt_data[0].cpu()), reconstructed_data)
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'image': helper.encode_image(reconstructed_data)
        }

    def process_custom_image(self, image, activation_function):
        try:
            image = image.resize((32,32), Image.LANCZOS)
            
            transform = transforms.ToTensor()
            input_tensor = transform(image).unsqueeze(0)
            
            result = self.process_single_image(0, activation_function)
            print(result)    
            return result
        
        except Exception as err:
            print(f"Error processing custom image: {err}")
    