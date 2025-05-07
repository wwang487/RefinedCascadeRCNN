import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config

from mmdet.apis import init_detector

# Load the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Hook to capture the features from a specific layer
def get_feature_maps(model, layer_name, x):
    features = None
    def hook(module, input, output):
        nonlocal features
        features = output
    layer = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook)
    model(x)
    handle.remove()
    return features

# Generate the heat map
def generate_heat_map(features):
    heat_map = features.squeeze(0).mean(dim=0).detach().cpu().numpy()
    heat_map = np.maximum(heat_map, 0)
    heat_map /= np.max(heat_map)
    return heat_map

# Visualize the heat map
def visualize_heat_map(image_path, heat_map, ramp = 'winter'):
    image = Image.open(image_path).convert('RGB')
    heat_map = np.uint8(255 * heat_map)
    heat_map = Image.fromarray(heat_map).resize(image.size, Image.ANTIALIAS)
    heat_map = np.array(heat_map)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(heat_map, cmap=ramp) # overlay heat map
    plt.axis('off')
    plt.show()

# Main process
def main_resnet(image_path, layer_name = 'layer4',ramp = 'winter'):
    model = models.resnet50(pretrained=True)
    model.eval()

    image = load_image(image_path)
    features = get_feature_maps(model, layer_name, image)
    heat_map = generate_heat_map(features)
    visualize_heat_map(image_path, heat_map, ramp = ramp)

def create_mmdet_model(cfg_file, checkpoint_file):
    cfg = Config.fromfile(cfg_file)
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    return model

def get_intermediate_outputs(model, img):
    model.eval()
    with torch.no_grad():
        # You may need to preprocess the image as per the model's requirement
        result = model.extract_feat(img)
        backbone_feat = result
        
        rpn_output = model.rpn_head(model.backbone(img))  # Example of extracting RPN output
        rpn_cls_score, rpn_bbox_pred = rpn_output

        # Assume the model is a Cascade RCNN; modify according to your model
        cascade_outputs = []
        rois = model.rpn_head.get_bboxes(rpn_cls_score, rpn_bbox_pred)
        for stage, bbox_head in enumerate(model.roi_head.bbox_head):
            rois = model.roi_head.bbox_roi_extractor(stage, backbone_feat, rois)
            cls_score, bbox_pred = bbox_head(rois)
            cascade_outputs.append((cls_score, bbox_pred))
            
        return rpn_output, cascade_outputs

