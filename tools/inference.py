import torchvision.transforms as t
from PIL import Image
import argparse
from cnn_mlp.model import get_model
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainer')

    parser.add_argument('--model_name', default="EfficientNetB3Pretrained")
    parser.add_argument('--model_path', default="")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--image_path', default="data/preprocessed_images/0_left.jpg")
    args = parser.parse_args()
    params = vars(args)
    print(params)

    model = get_model(args.model_name, args.device, {})
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    img_transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    classes = {
        0: "N",
        1: "D",
        2: "G",
        3: "C",
        4: "A",
        5: "H",
        6: "M",
        7: "O"
    }

    img = Image.open(args.image_path)
    img = img_transform(img).unsqueeze(0).to(args.device)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True).item()

    print("Прогноз : ", classes[pred])
