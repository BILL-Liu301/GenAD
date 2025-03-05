import cv2
import numpy as np
import argparse

def merge_videos(video_path1, video_path2, output_path):
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # 获取视频的帧率和尺寸
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确保两个视频的帧率一致，取较小的帧率
    fps = min(fps1, fps2, 3)

    # 创建一个视频写入对象，用于保存输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (max(width1, width2), height1 + height2))

    # 逐帧读取并拼接
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果任一视频结束，则退出循环
        if not ret1 or not ret2:
            break

        # 如果宽度不一致，调整宽度以匹配（可选，根据需要调整）
        if width1 != width2:
            frame1 = cv2.resize(frame1, (max(width1, width2), height1))
            frame2 = cv2.resize(frame2, (max(width1, width2), height2))

        # 拼接两个帧
        merged_frame = np.vstack((frame1, frame2))

        # 写入到输出视频
        out.write(merged_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    print(f"合并后的视频已保存到 {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Concat two videos')
    parser.add_argument('--video_path1', help='Path of the first video file')
    parser.add_argument('--video_path2', help='Path of the second video file')
    parser.add_argument('--output_path', help='Path of the output video file')
    args = parser.parse_args()

    return args

# 示例调用
video_path1 = "/workspace/genad/GenAD/visualization/testresults/results_origin.mp4"  # 第一个视频文件路径
video_path2 = "/workspace/genad/GenAD/visualization/testresults/results_change_cmd.mp4"  # 第二个视频文件路径
output_path = "/workspace/genad/GenAD/visualization/testresults/results.mp4"  # 输出视频文件路径

if __name__ == '__main__':
    args = parse_args()
    video_path1 = args.video_path1
    video_path2 = args.video_path2
    output_path = args.output_path

    # 合并视频
    merge_videos(video_path1, video_path2, output_path)
