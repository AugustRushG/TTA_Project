import json
import numpy as np
import cv2
import os


def convert_box(box_data, video_width, video_height):
    x = box_data['x'] / 100.0
    y = box_data['y'] / 100.0
    width = box_data['width'] / 100.0
    height = box_data['height'] / 100.0

    x_min = int(x * video_width)
    y_min = int(y * video_height)
    x_max = int((x + width) * video_width)
    y_max = int((y + height) * video_height)

    return [x_min, y_min, x_max, y_max]

def show_box_coordinates(box_coordinates, img_path):
    img = cv2.imread(img_path)
    x_min, y_min, x_max, y_max = box_coordinates
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow('Box', img)
    cv2.waitKey(0) # Wait for a key press to close the window
    cv2.destroyAllWindows()
    


def get_box_coordinates(box_information):

    close_scoreboard_box_x = box_information['box'][0]['sequence'][0]['x']
    close_scoreboard_box_y = box_information['box'][0]['sequence'][0]['y']
    close_scoreboard_box_width = box_information['box'][0]['sequence'][0]['width']
    close_scoreboard_box_height = box_information['box'][0]['sequence'][0]['height']

    close_box_data = {'x': close_scoreboard_box_x, 'y': close_scoreboard_box_y, 'width': close_scoreboard_box_width, 'height': close_scoreboard_box_height}

    far_scoreboard_box_x = box_information['box'][1]['sequence'][0]['x']
    far_scoreboard_box_y = box_information['box'][1]['sequence'][0]['y']
    far_scoreboard_box_width = box_information['box'][1]['sequence'][0]['width']
    far_scoreboard_box_height = box_information['box'][1]['sequence'][0]['height']
    far_box_data = {'x': far_scoreboard_box_x, 'y': far_scoreboard_box_y, 'width': far_scoreboard_box_width, 'height': far_scoreboard_box_height}

    return close_box_data, far_box_data


def main():

    pred_file = json.load(open('/home/august/github/TTA_Project/data/train_videos/scoreboard_data/scoreboard_reco_.json', 'r'))

    converted_data = []

    for video in pred_file:
        video_name = video['video']
        video_name = os.path.basename(video['video'])
        # remove file extension
        video_name = os.path.splitext(video_name)[0]
        # replace spaces with underscores
        video_name = video_name.replace(' ', '_')
        if '25WPF' in video_name:
            video_height = 720
            video_width = 1280
        else:
            video_height = 1080
            video_width = 1920
        print(f"Processing video: {video_name}")

        box_infor = video
        close_box_data, far_box_data = get_box_coordinates(box_infor)
        close_box_coordinates = convert_box(close_box_data, video_width, video_height)
        far_box_coordinates = convert_box(far_box_data, video_width, video_height)
        # show_box_coordinates(close_box_coordinates, '/home/august/github/TTA_Project/data/25WPE_SLO_M11_SF_Creange_FRA_v_von_Einem_AUS_game1_frames/009352.jpg')
        # show_box_coordinates(far_box_coordinates, '/home/august/github/TTA_Project/data/25WPE_SLO_M11_SF_Creange_FRA_v_von_Einem_AUS_game1_frames/009352.jpg')
        
        all_scores = video['left_score']
        close_scoreboard_scores = []
        far_scoreboard_scores = []
        for frame_score in all_scores:
            frame_index = frame_score['ranges'][0]['start'] - 1 # Convert to 0-based index
            score = frame_score['timelinelabels'][0]
            if score.startswith('c'):
                real_score = score[1:]
                close_scoreboard_scores.append({'video_name': video_name, 'frame_index': frame_index, 'score': int(real_score), 'box_coordinates': close_box_coordinates})

            elif score.startswith('f'):
                real_score = score[1:]
                far_scoreboard_scores.append({'video_name': video_name, 'frame_index': frame_index, 'score': int(real_score), 'box_coordinates': far_box_coordinates})
        
        converted_data.extend(close_scoreboard_scores)
        converted_data.extend(far_scoreboard_scores)
    
    with open('/home/august/github/TTA_Project/data/train_videos/scoreboard_data/converted_scoreboard_data.json', 'w') as f:
        json.dump(converted_data, f, indent=4)
        




if __name__ == "__main__":
    main()
