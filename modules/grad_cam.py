import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()
            
        # Target score
        loss = output[0, target_category]
        
        # Backward pass
        loss.backward()
        
        # Pull gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of feature maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU to retain only features that positively correlate with target category
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, target_category

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def generate_gradcam_heatmap(model, input_image, input_tensor, transform_func=None):
    """
    Generates a beautiful Grad-CAM heatmap visualization overlayed on the input PIL image.
    Works for PyTorch model.
    """
    try:
        # Check if PyTorch model
        if not isinstance(model, torch.nn.Module):
            return None
            
        # EfficientNet-B0 final features block: model.features[-1] or model.features
        target_layer = None
        if hasattr(model, 'features'):
            # For EfficientNet, features[-1] is the last block
            target_layer = model.features[-1]
            
        if target_layer is None:
            # Fallback to any convolutional module we can find
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
                    
        if target_layer is None:
            return None
            
        cam_generator = GradCAM(model, target_layer)
        
        # Generate raw CAM
        cam, category_idx = cam_generator(input_tensor, target_category=None)
        cam_generator.remove_hooks()
        
        # Process heatmap
        img_w, img_h = input_image.size
        heatmap = cv2.resize(cam, (img_w, img_h))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply JET colormap for vibrant representation
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        orig_np = np.array(input_image.convert('RGB'))
        
        # Superimpose
        superimposed = heatmap_color * 0.4 + orig_np * 0.6
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(superimposed), category_idx
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None
