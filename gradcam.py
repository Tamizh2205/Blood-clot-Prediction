"""
gradcam.py
──────────
Generates Grad-CAM heatmap overlaid on medical images.
Works with EfficientNet (CT/Ultrasound) and ResNet50 (MRI).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class GradCAM:
    """
    Grad-CAM implementation for binary classifiers.
    Works with EfficientNet and ResNet architectures.
    """

    def __init__(self, model, model_type='efficientnet'):
        self.model      = model
        self.model_type = model_type
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Pick the last conv layer based on architecture
        if self.model_type == 'efficientnet':
            # timm EfficientNet last conv block
            target_layer = self.model.conv_head
        elif self.model_type == 'resnet':
            # torchvision ResNet last conv layer
            target_layer = self.model.layer4[-1].conv2
        else:
            target_layer = list(self.model.modules())[-3]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, image_input) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        Returns: np.ndarray — heatmap overlaid on original image (RGB, 0-255)
        """
        # Preprocess image
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(
                cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                if image_input.shape[2] == 3 else image_input
            )
        else:
            img_pil = image_input.convert('RGB')

        # Keep original for overlay
        img_orig = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))

        # Forward pass
        tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_()

        self.model.eval()
        output = self.model(tensor).squeeze(1)
        prob   = torch.sigmoid(output)

        # Backward pass for target class
        self.model.zero_grad()
        output.backward()

        # Compute Grad-CAM
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]

        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to image size
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        # Apply colormap
        heatmap_uint8  = np.uint8(255 * heatmap_resized)
        heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color  = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        superimposed = cv2.addWeighted(img_orig, 0.55, heatmap_color, 0.45, 0)

        return superimposed, float(prob.item())


def generate_gradcam(image_input, model, model_type='efficientnet'):
    """
    Convenience function — returns overlaid heatmap image as PIL Image.
    """
    cam    = GradCAM(model, model_type)
    result, prob = cam.generate(image_input)
    return Image.fromarray(result), prob