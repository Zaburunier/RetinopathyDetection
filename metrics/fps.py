from model import get_model
import torch
from tqdm import tqdm
import argparse


def fps_test(args_):
    time_list = []

    inp = torch.randn(1, 3, 224, 224).to(args_.device)
    model = get_model(args_.model_name, args_.device, {})

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in tqdm(range(args_.iter_count)):

        start.record()
        _ = model(inp)
        end.record()

        # Ждет, пока все завершится
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000
        if i > args_.warmup:
            time_list.append(sec)

    avg_time = sum(time_list) / len(time_list)
    avg_fps = 1 / avg_time
    print(f"Название модели: {args_.model_name} Среднее время: {avg_time} Средний FPS: {avg_fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FPS')
    parser.add_argument('--model_name', default="MobileNetV2Pretrained", help='You can use all models in model.py')
    parser.add_argument('--warmup', default=20, help='warmup')
    parser.add_argument('--iter_count', default=100, help='iter_count')
    parser.add_argument('--device', default="cuda", help='device')
    args = parser.parse_args()
    print(args)

    fps_test(args)
