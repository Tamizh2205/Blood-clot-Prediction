import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_ultrasound_model(model_path: str = 'models/ultrasound_model.pth'):
    model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def predict_ultrasound(image_input, model) -> dict:
    """
    Predict DVT clot from ultrasound image.
    image_input: file path (str) or PIL Image or numpy array
    Returns dict: {label, confidence, risk_level}
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(image_input)
    else:
        img = image_input

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor).squeeze(1)
        prob   = torch.sigmoid(output).item()

    label      = 'DVT Clot Detected' if prob >= 0.5 else 'No DVT Clot Detected'
    risk_level = 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= 0.5 else 'LOW')

    return {
        'label'      : label,
        'confidence' : round(prob * 100, 1),
        'risk_level' : risk_level,
        'raw_prob'   : prob
    }