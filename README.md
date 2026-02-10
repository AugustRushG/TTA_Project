# TTA Project - Table Tennis Analysis

A computer vision project for analyzing table tennis videos, featuring event detection and ball tracking capabilities.

## Features

- **Event Detection**: Automatically detect various table tennis events including:
  - Ball bounces on table (far/close)
  - Forehand/backhand shots (far/close)
  - Serve detection (far/close)
- **Ball Tracking**: Track ball movement throughout the video
- **Table Detection**: Identify and analyze table regions
- **Visualization**: Generate visual outputs showing detected events and ball trajectories

## Project Structure

```
TTA_Project/
├── data/                          # Video files and extracted frames
├── src/                           # Source code
│   ├── ball_tracking/            # Ball tracking models and utilities
│   ├── event_detection/          # Event detection models
│   ├── input_process/            # Video and frame processing
│   ├── table_detector/           # Table detection utilities
│   ├── utils/                    # Utility functions and visualization
│   └── main.py                   # Main execution script
├── environment.yml               # Conda environment specification
├── requirements.txt              # Python package requirements
└── README.md                     # This file
```

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- CUDA-compatible GPU (recommended for faster processing)

### Setup Environment

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TTA_Project
   ```

2. **Create and activate the conda environment:**
   ```bash
   # Create environment from environment.yml
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate TTA

   If you perfer not to use conda:
   # Create a virtual environment named venv
   python -m venv venv
  
   # Activate the environment (PowerShell)
   .\venv\Scripts\Activate.ps1
  
   # OR activate (Command Prompt)
   venv\Scripts\activate.bat
  
   # OR activate (macOS/Linux)
   source venv/bin/activate
   
   ```


4. **Alternative installation using requirements.txt:**
   ```bash
   # Create a new conda environment with Python 3.11
   conda create -n TTA python=3.11
   conda activate TTA
   
   # Install required packages
   pip install -r requirements.txt
   
   # Install additional conda packages for video processing
   conda install -c conda-forge ffmpeg
   ```

5. **Download all pretrained models from the following link:**
   Google Drive: https://drive.google.com/drive/folders/1pMVfLtjA1zKmJAYUPRszMQ2vKuRQNlf0?usp=sharing
   After downloading put them in the corresponding folder
   

### Verify Installation

```bash
# Check if PyTorch is installed with CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## Usage

### Prepare Your Data

1. **Place your video files** in the `data/` directory:
   ```bash
   cp your_video.mp4 data/
   ```

2. **Supported video formats**: MP4, AVI, MOV, etc. (any format supported by OpenCV)

### Run Analysis

Execute the main script with your video file:

```bash
cd src
python main.py --video_path ../data/your_video.mp4
```

### Command Line Options

```bash
python main.py --help
```

**Available parameters:**
- `--video_path`: Path to the input video file (required)
- `--window_size`: Size of the sliding window for frame processing (default: 100)
- `--stride`: Stride for the sliding window (default: 50)
- `--device`: Device to run models on - 'cpu', 'cuda', or 'mps' (default: 'cuda')

### Example Commands

```bash
# Basic usage
python main.py --video_path ../data/match_video.mp4

# Custom window parameters
python main.py --video_path ../data/match_video.mp4 --window_size 150 --stride 75

# Force CPU usage (if GPU not available)
python main.py --video_path ../data/match_video.mp4 --device cpu
```

## Output

The analysis will generate:

1. **Frame extraction**: Individual frames saved in `data/<video_name>_frames/`
2. **Event predictions**: JSON file with detected events `predicted_events_<video_name>.json`
3. **Visualizations**: 
   - `bounces_on_table_split.jpg`: Visual representation of detected bounces
   - Additional visualization files in the `result/` directory

### Output Format

The predicted events JSON contains:
```json
{
  "events": [
    {
      "frame": 123,
      "timestamp": 4.1,
      "event_type": "far_table_bounce",
      "confidence": 0.95
    }
  ]
}
```

## Event Types

| Code | Event Type |
|------|------------|
| 0 | empty |
| 1 | far_table_bounce |
| 2 | far_table_forehand |
| 3 | far_table_backhand |
| 4 | far_table_serve |
| 5 | close_table_bounce |
| 6 | close_table_forehand |
| 7 | close_table_backhand |
| 8 | close_table_serve |

## System Requirements

- **Python**: 3.11
- **GPU Memory**: 4GB+ VRAM recommended
- **RAM**: 8GB+ system RAM
- **Storage**: Sufficient space for frame extraction (varies by video length)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `window_size` or use `--device cpu`
2. **OpenCV video reading errors**: Ensure video file is not corrupted and codec is supported
3. **Import errors**: Verify all packages are installed in the correct environment

### Performance Tips

- Use GPU (`--device cuda`) for faster processing
- Adjust `window_size` and `stride` based on your system capabilities
- Process shorter video clips for testing before running on full videos

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.
