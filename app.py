# app.py
import base64
from io import BytesIO
import time
from flask import Flask, json, jsonify, redirect, render_template, request, session, url_for
from flask_socketio import SocketIO, emit
from flask_session import Session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import secrets


app = Flask(__name__)
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
)
socketio = SocketIO(app)

# Initialize CIFAR and transforms
dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()
if torch.cuda.is_available():
    device = "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def process_single_image(idx, activation_function):
    helper = Helper()
    net = LeNet(activation_function).to(device)
    net.apply(helper.weights_init)
    criterion = helper.cross_entropy_for_onehot
    
    # Process the image
    gt_data = tp(dst[idx][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[idx][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = helper.label_to_onehot(gt_label)
    
    # Compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    
    # Generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
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
    reconstructed_data = tt(dummy_data[0].cpu())
    mse, psnr, ssim = helper.check_similarity(tt(gt_data[0].cpu()), reconstructed_data)
    
    # Convert the reconstructed image to base64 for display
    buffer = BytesIO()
    reconstructed_data.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'image': img_str
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/handle_data_single', methods=['POST'])
def handle_data_single():
    cifar_index = int(request.form['cifar_index'])
    activation_function = request.form['activation_function']
    return render_template("loading_single.html", cifar_index=cifar_index, activation_function=activation_function)

@socketio.on('start_single_process')
def handle_single_process(data):
    cifar_index = data['cifar_index']
    activation_function = data['activation_function']
    try:
        result = process_single_image(cifar_index, activation_function)
        emit('complete',{'result':result})
    except Exception as err:
        emit('error', str(err))
        
@app.route('/result')
def result():
    try:
        data = json.loads(request.args.get('data'))
        cifar_index = data.get('cifar_index',0) 
        
        original_img = dst[cifar_index][0]
        buffer = BytesIO()
        original_img.save(buffer, format='PNG')
        original_img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return render_template('result.html', result=data, cifar_index=cifar_index, original_image=original_img_str)
    except Exception as err:
        print(f"Error at: {err}")
        return redirect(url_for('index'))
    
@app.route('/handle_data_multiple', methods=['POST'])
def handle_data_multiple():
    start_cifar_index = int(request.form['start_cifar_index'])
    end_cifar_index = int(request.form['end_cifar_index'])
    activation_function = request.form['activation_function']
    
    return render_template("loading_multiple.html", start_cifar_index=start_cifar_index, end_cifar_index=end_cifar_index, activation_function=activation_function)

@socketio.on('start_processing')
def handle_process(data):
    try:
        start_cifar_index = data['start_index']
        end_cifar_index = data['end_index']
        activation_function = data['activation_function']
        total = end_cifar_index - start_cifar_index + 1
        
        results = []
        for i, cifar_id in enumerate(range(start_cifar_index, end_cifar_index + 1)):
            result = process_single_image(cifar_id, activation_function)
            results.append({'cifar_id': cifar_id, **result})
            
            progress = ((i + 1)/total) * 100
            emit('progress', {
                'progress': progress,
                'current_result': result
            })
            time.sleep(0.1)
        
        # Send results directly to client
        emit('complete', {'results': results})
        
    except Exception as err:
        print(f"Processing error: {err}")
        emit('error', str(err))

@app.route('/chart')
def chart():
    try:
        results_json = request.args.get('results')
        if not results_json:
            return redirect(url_for('index'))
            
        results = json.loads(results_json)
        
        # Add original images
        for result in results:
            cifar_id = result['cifar_id']
            original_img = dst[cifar_id][0]
            buffer = BytesIO()
            original_img.save(buffer, format='PNG')
            result['original_image'] = base64.b64encode(buffer.getvalue()).decode()
        
        return render_template('chart_multiple.html', results=results)
    except Exception as e:
        print(f"Chart error: {e}")
        return redirect(url_for('index'))



if __name__ == '__main__':
    socketio.run(app, debug=True)
