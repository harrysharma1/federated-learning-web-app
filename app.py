# app.py
import base64
from io import BytesIO
from random import Random
import time
from flask import Flask, json, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
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
    
    def encode_image(self, image_data):
        buffer = BytesIO()
        image_data.save(buffer, format='PNG')
        return base64.b85encode(buffer.getvalue()).decode()
    
    def decode_image(self, b85_string):
        return base64.b85decode(b85_string)
    
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
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'image': helper.encode_image(reconstructed_data)
    }

@app.route("/")
def index():
    return render_template("index.html")


# Single Choice

@app.route('/handle_data_single', methods=['GET','POST'])
def handle_data_single():
    if request.method == 'POST':
        cifar_index = int(request.form['cifar_index'])
        activation_function = request.form['activation_function']

        return render_template("loading_single.html", cifar_index=cifar_index, activation_function=activation_function)
    else:
        return redirect(url_for('index'))

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
    helper = Helper()
    try:
        # Get and validate data
        data_str = request.args.get('data')
        if not data_str:
            raise ValueError("No data provided")
            
        data = json.loads(data_str)
        cifar_index = data.get('cifar_index')
        if cifar_index is None:
            raise ValueError("No CIFAR index provided")
        
        # Get and encode original image
        original_img = dst[cifar_index][0]
        data['original_image'] = helper.encode_image(original_img)
        
        # Validate reconstructed image exists
        if 'image' not in data:
            raise ValueError("No reconstructed image in data")

        return render_template('result.html', result=data)
    except Exception as err:
        print(f"Error in result route: {err}")
        print(f"Request args: {request.args}")
        return redirect(url_for('index'))
# Single Random

@app.route('/handle_data_single_random', methods=['GET', 'POST'])
def  handle_data_single_random():
    if request.method == 'POST':
        random = Random()
        cifar_index = random.randint(0,49999)
        activation_function = secrets.choice(['relu','sigmoid','tanh'])
        return render_template("loading_single.html", cifar_index=cifar_index, activation_function=activation_function) 
    else:
        return redirect(url_for('index'))

# Multiple Choices

@app.route('/handle_data_multiple', methods=['GET','POST'])
def handle_data_multiple():
    if request.method == 'POST':
        start_cifar_index = int(request.form['start_cifar_index'])
        end_cifar_index = int(request.form['end_cifar_index'])
        activation_function = request.form['activation_function']
        
        return render_template("loading_multiple.html", start_cifar_index=start_cifar_index, end_cifar_index=end_cifar_index, activation_function=activation_function)
    else:
        return redirect(url_for('index'))
    
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
                'current_result': result,
                'curr_id': cifar_id
            })
            time.sleep(0.1)
        
        # Send results directly to client
        emit('complete', {'results': results})
        
    except Exception as err:
        print(f"Processing error: {err}")
        emit('error', str(err))

@socketio.on('convert_image')
def convert_image(data):
    helper = Helper()
    try:
        base85_str = data['image']
        bytes_data = helper.decode_image(base85_str)
        base64_str = base64.b64encode(bytes_data).decode()
        emit('image_converted', {
            'image' : base64_str
        }) 
    except Exception as err:
        emit ('error', str(err))

@app.route('/chart')
def chart():
    helper = Helper()
    try:
        results_json = request.args.get('results')
        if not results_json:
            return redirect(url_for('index'))
            
        results = json.loads(results_json)
        
        # Add original images
        for result in results:
            cifar_id = result['cifar_id']
            original_img = dst[cifar_id][0]
            result['original_image'] = helper.encode_image(original_img)
        
        return render_template('chart_multiple.html', results=results)
    except Exception as e:
        print(f"Chart error: {e}")
        return redirect(url_for('index'))

# Multiple Random

@app.route('/handle_data_random_range', methods=['GET', 'POST'])
def handle_data_random_range():  # Changed function name to match URL
    if request.method == 'POST':
        # Generate random range between 1-15 images
        random = Random()
        range_size = random.randint(1, 15)
        start_index = random.randint(0, 49999 - range_size)
        end_index = start_index + range_size - 1
        activation_function = random.choice(['relu', 'sigmoid', 'tanh'])
        
        return render_template("loading_multiple.html", 
                             start_cifar_index=start_index,
                             end_cifar_index=end_index,
                             activation_function=activation_function)
    else:
        return redirect(url_for('index'))
# Encode Decode Image

@app.template_filter('b85decode')
def b85decode_filter(b85_string):
    """Convert base85 to bytes"""
    return base64.b85decode(b85_string)

@app.template_filter('b64encode')
def b64encode_filter(data):
    """Convert bytes to base64"""
    return base64.b64encode(data).decode()

if __name__ == '__main__':
    socketio.run(app, debug=True)
