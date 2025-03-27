import cv2
import numpy as np
from datetime import datetime
import os
import VideoData


def get_MinMax_frameNb(videoData: VideoData.VideoData) -> tuple[int, int]:
    '''
    Convertit les timestamps en frames.
    '''
    if videoData.t2 is not None:
        max_frame = int(videoData.t2 * videoData.fps)
    else:
        max_frame = np.inf
    min_frame = int(videoData.t1 * videoData.fps)
    return min_frame, max_frame


def get_res(videoData: VideoData.VideoData) -> tuple[int, int]:
    '''
    Renvoie la résolution de l'image. Attention à la rotation.
    '''
    width = int(videoData.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoData.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if videoData.rotate:
        res = (height, width)
    else:
        res = (width, height)
    return res


def generate_file_name(info: str, ext: str = "npz") -> str:
    now = datetime.now()
    date_str = now.strftime("%d-%m")
    time_str = now.strftime("%H-%M-%S")
    return f"{date_str}_{time_str}_{info}.{ext}"


def generate_file_name(videoData: VideoData.VideoData, ext: str = "npz"):
    timestamp = os.path.getctime('test.mp4')
    date_time = datetime.fromtimestamp(timestamp)
    date_str = date_time.strftime('%d-%m')

    # time_str = now.strftime("%H-%M-%S")
    return f"{date_str}_h{str(videoData.height).replace(".", "-")}_m{videoData.mass}{videoData.info}.{ext}"
