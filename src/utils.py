from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import torch
import numpy as np
import torch.nn.functional as F
import base64
from io import BytesIO

class Helper():
    """Helper class for utility functions.     
    """
    
    def label_to_onehot(self, target, num_classes=100):
        """Converting class labels to one-hot encoded vectors.<br>
        Taken from DLG: https://github.com/mit-han-lab/dlg/blob/master/utils.py

        Args:
            target (torch.Tensor): Tensor containing class labels with shape (N,), where N is the no. of classes.
            num_classes (int, optional): _description_. Total number of classes to 100.

        Returns:
            torch.Tensor: A tensor of shape (N, num_classes) containing one-hote encoded representation of input labels.
        """
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def cross_entropy_for_onehot(self, pred, target):
        """Computes the cross-entropy loss for the one-hot encoded targets.<br>
        Taken from DLG: https://github.com/mit-han-lab/dlg/blob/master/utils.py

        Args:
            pred (torch.Tensor): Predicted probabilities with shape (N, C), where N is batch size and C is no. of classes.
            target (torch.Tensor): The one-hot encoded target tensor with shape (N,C), where N and C are same as pred.

        Returns:
            torch.Tensor: The computed cross-entropy loss as a scalar tensor.
        """
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

    def weights_init(self, m):
        """Initialises the weights and bias of a given model, in this case LeNet.<br>
        Taken from DLG: https://github.com/mit-han-lab/dlg/blob/master/models/vision.py

        Args:
            m (nn.Module): _description_
        """
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    
    def check_similarity(self, ground_truth, reconstructed):
        """Computes similarity metrics between ground truth and the reconstructed image.

        Args:
            ground_truth (PIL.Image): Original image to compare against the reconstructed image from gradient.
            reconstructed (PIL.Image): Image reconstructed from DLG algorithm to compare.

        Returns:
            tuple: Tuple containing three floats
                - mse (float): Mean Squared Error between each image.
                - psnr (float): Peak-Signal-to-Noise ratio, measured in decibels (dB).
                - ssim (float): Structural Similarity Index Measure (in the range 0.0<=x<=1.0).
        """
        ground_truth = np.array(ground_truth)
        reconstructed = np.array(reconstructed)
        
        ground_truth = (ground_truth * 255).astype(np.uint8)
        reconstructed = (reconstructed * 255).astype(np.uint8)
        
        mse = mean_squared_error(ground_truth, reconstructed)
        psnr = peak_signal_noise_ratio(ground_truth, reconstructed)
        ssim, _ = structural_similarity(ground_truth, reconstructed, full=True, win_size=3)
        
        return mse, psnr, ssim
    
    def encode_image(self, image_data):
        """Encoding image into a Base85 string.

        Args:
            image_data (PIL.Image): The image to be encoded.

        Returns:
            str: Base85-encoded string representation of the image in PNG format.
        """
        buffer = BytesIO()
        image_data.save(buffer, format='PNG')
        return base64.b85encode(buffer.getvalue()).decode()
    
    def decode_image(self, b85_string):
        """Decoding Base85-encoded string into the Image.

        Args:
            b85_string (str): A Base85-encoded string.

        Returns:
            PIL.Image: The decoded Image.
        """
        return base64.b85decode(b85_string)
    
class LocalSession():
    """Local Session Class.
    """
    def __init__(self):
        """Initialise and empty list for use of local session in WebSocket.
        """
        self.results=[]
    
    def add(self, item):
        """Add dictionary item to transfer data through WebSocket.

        Args:
            item (Dict): Key-Value pair to be transferred through WebSocket. 
        """
        self.results.append(item)

    def get_results(self):
        """Return full list of values.

        Returns:
            List[Dict]: List of dictionary items.
        """
        return self.results
    
    def remove(self, item):
        """Remove an item from the list.

        Args:
            item (Dict): Remove the dictionary item specificed in function parameters.
        """
        self.results.remove(item)
        
    def clear(self):
        """Clear local session storage.
        """
        self.results = []
