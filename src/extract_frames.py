import argparse
import os
from input_process.video_loader import extract_frames

def parse_args():
    parser = argparse.ArgumentParser(description="Process video frames for event detection and ball tracking.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file or folder.')
    parser.add_argument('--frame_dir', type=str, default=None, help='Directory to save extracted frames. If not provided, frames will be saved in a directory with the same name as the video.')
    return parser.parse_args()



def main():
    args = parse_args()
    # if video_path is a folder, extract frames from all videos in the folder
    if os.path.isdir(args.video_path):
        for filename in os.listdir(args.video_path):
            if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mkv"):
                video_file = os.path.join(args.video_path, filename)
                frame_dir = args.frame_dir if args.frame_dir else os.path.splitext(video_file)[0]
                extract_frames(video_path=video_file, frame_path=frame_dir) # dont resize
    else:
        extract_frames(video_path=args.video_path, frame_path=args.frame_dir) # dont resize


if __name__ == "__main__":
    main()


