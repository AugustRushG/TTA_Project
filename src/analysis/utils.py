import json


class analysis_utils:
    @staticmethod
    def load_json(file_path):
        """
        Load a JSON file and return its content.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            dict: Parsed JSON content.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_path):
        """
        Save data to a JSON file.
        
        Args:
            data (dict): Data to save.
            file_path (str): Path where the JSON file will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def count_bounces(input_data):
        """
        Count the number of bounces in the input JSON data.
        
        Args:
            input_json (str): Path to the input JSON file containing event data.
            
        Returns:
            int: Total number of bounces detected in the video.
        """
        
        close_table_bounces = 0
        far_table_bounces = 0
        for frame_idx, frame_info in input_data.items():
            if 'event_type' not in frame_info:
                continue
            event_type = frame_info['event_type']
            if event_type == 'close_table_bounce':
                close_table_bounces += 1
            elif event_type == 'far_table_bounce':
                far_table_bounces += 1
        print(f"Total close table bounces: {close_table_bounces}")
        print(f"Total far table bounces: {far_table_bounces}")
        
        return close_table_bounces + far_table_bounces, close_table_bounces, far_table_bounces
    
    def count_serves(input_data):
        """
        Count the number of serves in the input JSON data.
        
        Args:
            input_json (str): Path to the input JSON file containing event data.
            
        Returns:
            int: Total number of serves detected in the video.
        """
        
        far_table_serve = 0
        close_table_serve = 0
        for frame_idx, frame_info in input_data.items():
            if 'event_type' not in frame_info:
                continue
            event_type = frame_info['event_type']
            if event_type == 'far_table_serve':
                far_table_serve += 1
            elif event_type == 'close_table_serve':
                close_table_serve += 1
        print(f"Total far table serves: {far_table_serve}")
        print(f"Total close table serves: {close_table_serve}")
        
        return far_table_serve + close_table_serve, far_table_serve, close_table_serve

    def calculate_points(input_data):
        """
        Calculate the points scored based on the input JSON data.
        
        Args:
            input_json (str): Path to the input JSON file containing event data.
            
        Returns:
            int: Total points scored in the video.
        """
        input_data = list(input_data.values())
        close_table_points = 0
        far_table_points = 0
        for i in range(len(input_data)):
            if 'event_type' not in input_data[i]:
                continue
            event_type = input_data[i]['event_type']
            
            if i > 0:
                j = i - 1
                while j >= 0 and not input_data[j]['event_type'].endswith('_bounce'):
                    j -= 1
                previous_event = input_data[j]['event_type'] if j >= 0 else None
            else:
                previous_event = None

            if event_type == 'close_table_serve' or event_type == 'far_table_serve':
                if previous_event == 'close_table_bounce':
                    far_table_points += 1
                elif previous_event == 'far_table_bounce':
                    close_table_points += 1
                
        print(f"Total close table points: {close_table_points}")
        print(f"Total far table points: {far_table_points}")
        return close_table_points + far_table_points, close_table_points, far_table_points