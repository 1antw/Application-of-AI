import argparse
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as trns
from PIL import Image


def run_image_classification(model, image_path, transforms, classes, topk=5):
    """Inference
    """
    # Read image and run prepro
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image)
    print(f"\n\nImage size after transformation: {image_tensor.size()}")

    image_tensor = image_tensor.unsqueeze(0)
    print(f"Image size after unsqueezing: {image_tensor.size()}")

    # Feed input
    output = model(image_tensor)
    print(f"Output size: {output.size()}")

    output = output.squeeze()
    print(f"Output size after squeezing: {output.size()}")

    # Result postpro
    _, indices = torch.sort(output, descending=True)
    probs = F.softmax(output, dim=-1)

    print("\n\nInference results:")
    for index in indices[:topk]:
        print(f"Label {index}: {classes[index]} ({probs[index].item():.2f})")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PyTorch Image Classification")
    parser.add_argument("--image_path", type=str,
                        default="C:\\Users\\iam1a\\Downloads\\maltese.jpeg", help="path to image")
    parser.add_argument("--class_def", type=str,
                        default="C:\\Users\\iam1a\\Downloads\\imagenet_classes.txt", help="path to ImageNet class definition")

    # Parse arguments
    args = parser.parse_args()

    print(dir(models))

    # Load ImageNet classes
    with open(args.class_def) as f:
        classes = [line.strip() for line in f.readlines()]

    # Define image transforms
    transforms = trns.Compose([trns.Resize((224, 224)), trns.ToTensor(), trns.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load model
    mobilenet_v2_model = models.mobilenet_v2(pretrained=True)
    print(mobilenet_v2_model)

    # Set model to eval mode
    mobilenet_v2_model.eval()

    # Run model
    with torch.no_grad():
        run_image_classification(
            mobilenet_v2_model, args.image_path, transforms, classes)
