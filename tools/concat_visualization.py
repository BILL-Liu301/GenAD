import argparse
import os
import cv2
import json
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Concat Images of Visualization')
    parser.add_argument('--visualization_path', help='Path of the visualization folder')
    parser.add_argument('--save_path', help='Path of the output video file')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # 提取modes和sample_tokens
    modes = sorted(os.listdir(args.visualization_path))
    sample_tokens = set()
    num_samples = 0
    for mode in modes:
        for f in os.listdir(os.path.join(args.visualization_path, mode)):
            if f.endswith('.png'):
                sample_tokens.add(f.split('.')[0])
                num_samples += 1
        assert num_samples == len(sample_tokens)
        num_samples = 0

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 按照sample_token进行合并图片,并提取出各图片
    for sample_token in tqdm(sample_tokens):
        img_list = []
        for mode in modes:
            img_path = os.path.join(args.visualization_path, mode, sample_token + '.png')
            img_list.append(cv2.imread(img_path))
        img_concat = cv2.vconcat(img_list)
        cv2.imwrite(os.path.join(args.save_path, sample_token + '.png'), img_concat)

        # 复制其他文件
        os.makedirs(os.path.join(args.save_path, sample_token))
        shutil.copy(
            os.path.join(args.visualization_path, modes[0], 'samples', f'CAMS_{sample_token}.png'),
            os.path.join(args.save_path, sample_token, f'cams.png')
        )
        shutil.copy(
            os.path.join(args.visualization_path, modes[0], 'samples', f'des_{sample_token}.json'),
            os.path.join(args.save_path, sample_token, f'description.json')
        )
        for mode in modes:
            shutil.copy(
                os.path.join(args.visualization_path, mode, 'samples', f'bev_pred_{sample_token}.png'),
                os.path.join(args.save_path, sample_token, f'{mode}.png')
            )
            shutil.copy(
                os.path.join(args.visualization_path, mode, 'samples', f'bev_pred_map_{sample_token}.png'),
                os.path.join(args.save_path, sample_token, f'{mode}_plot_bev_pred_map.png')
            )
            shutil.copy(
                os.path.join(args.visualization_path, mode, 'samples', f'bev_pred_agents_{sample_token}.png'),
                os.path.join(args.save_path, sample_token, f'{mode}_plot_bev_pred_agents_.png')
            )
