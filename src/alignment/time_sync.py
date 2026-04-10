# src/alignment/time_sync.py
import re
import scipy.io
import h5py
import os
import cv2
import pandas as pd
from datetime import datetime, timedelta


def convert_to_datetime(arr):
    return datetime(
        int(arr[0]),
        int(arr[1]),
        int(arr[2]),
        int(arr[3]),
        int(arr[4]),
        int(arr[5])
    )

def mat_struct_to_dict(obj):
    """
    Recursively convert mat_struct to dict
    """
    if isinstance(obj, list):
        return [mat_struct_to_dict(o) for o in obj]

    if hasattr(obj, "__dict__"):
        result = {}
        for key in obj.__dict__.keys():
            if not key.startswith("_"):
                result[key] = mat_struct_to_dict(obj.__dict__[key])
        return result

    return obj

def load_mat_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        data = {k: v for k, v in data.items() if not k.startswith("__")}

        # convert everything
        data = {k: mat_struct_to_dict(v) for k, v in data.items()}

        return data

    except NotImplementedError:
        with h5py.File(file_path, 'r') as f:
            return {k: f[k][()] for k in f.keys()}

# -----------------------------
# 1. Parse video start time
# -----------------------------
def parse_video_start_time(filename):
    """
    Extract timestamp from filename.

    Expected format:
    monkey_20230801_10-00-00.mp4

    Returns:
        datetime object
    """
    pattern = r"(\d{8})_(\d{2}_\d{2}_\d{2})"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Could not parse time from filename: {filename}")

    date_part = match.group(1)  # YYYYMMDD
    time_part = match.group(2).replace("_", ":")  # HH:MM:SS

    return datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H:%M:%S")

# -----------------------------
# 2. Convert behavior time → frame
# -----------------------------
def get_event_time_from_trial(trial_datetime, code_times, code_numbers, target_code=40):
    """
    Extract event time (absolute datetime) for given code.
    
    code_times: in milliseconds
    """
    for t, c in zip(code_times, code_numbers):
        if c == target_code:
            return trial_datetime + timedelta(milliseconds=t)
    
    raise ValueError(f"Code {target_code} not found")


def compute_event_frame(event_time, video_start_time, fps):
    """
    Convert event timestamp to frame index.

    Args:
        event_time (datetime)
        video_start_time (datetime)
        fps (float)

    Returns:
        int: frame index
    """
    delta = event_time - video_start_time
    seconds = delta.total_seconds()

    if seconds < 0:
        raise ValueError("Event occurs before video start")

    return int(seconds * fps)


# -----------------------------
# 3. Get frame window
# -----------------------------
def get_frame_window(event_frame, fps, window_sec=1.0):
    """
    Get frame window around event.

    Args:
        event_frame (int)
        fps (float)
        window_sec (float)

    Returns:
        (start_frame, end_frame)
    """
    start = event_frame
    end = int(event_frame + window_sec * fps)
    return start, end


def get_trial_frames_from_behavior(video_start_time, trial_datetime,
                                    code_times, code_numbers, fps, window_sec=1.0,target_code = 40):
    
    event_time = get_event_time_from_trial(
        trial_datetime,
        code_times,
        code_numbers,
        target_code
    )

    event_frame = compute_event_frame(event_time, video_start_time, fps)
    start, end = get_frame_window(event_frame, fps, window_sec)

    return {
        "event_time": event_time,
        "event_frame": event_frame,
        "start_frame": start,
        "end_frame": end
    }




def Behavior_parser(file_path, video_start_time, fps, window_sec=1.0, target_code=40):

    mat_data = load_mat_file(file_path)

    trials = mat_data["dataSel"]

    rows = []

    for i, trial in enumerate(trials):

        try:
            trial_number = trial.Trial
            trial_datetime_arr = trial.TrialDateTime
            trial_datetime = convert_to_datetime(trial_datetime_arr)


            code_times = trial.BehavioralCodes.CodeTimes
            code_numbers = trial.BehavioralCodes.CodeNumbers

            trial_info = None
            if hasattr(trial, "UserVars") and hasattr(trial.UserVars, "trialInfo"):
                trial_info = trial.UserVars.trialInfo

            # -------------------------
            # Trial start frame
            # -------------------------
            trial_start_frame = compute_event_frame(
                trial_datetime,
                video_start_time,
                fps
            )

         
            event_time = get_event_time_from_trial(
                trial_datetime,
                code_times,
                code_numbers,
                target_code
            )

            event_frame = compute_event_frame(
                event_time,
                video_start_time,
                fps
            )

            # -------------------------
            # Window
            # -------------------------
            start_frame, end_frame = get_frame_window(
                event_frame,
                fps,
                window_sec
            )

            # -------------------------
            # Store row
            # -------------------------
            rows.append({
                "trial_index": i,
                "trial_number": trial_number,
                "trial_datetime": trial_datetime,
                "trial_start_frame": trial_start_frame,
                "event_time": event_time,
                "event_frame": event_frame,
                "end_frame": end_frame,
                "trial_info": trial_info
            })

        except Exception as e:
            print(f"Skipping trial {i}: {e}")
            continue

    return pd.DataFrame(rows)



def get_video_info(video_path):
    """
    Extract video metadata (fps, frame count, duration)
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        raise ValueError("FPS is zero — video may be corrupted")

    duration = total_frames / fps

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration
    }