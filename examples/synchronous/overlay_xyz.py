import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 強制使用 Agg 來避免 Qt 介面錯誤
matplotlib.use("Agg")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.common_models import Camera
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient
from ganzin.sol_sdk.utils import find_nearest_timestamp_match

# 初始化數據緩衝區
buffer_size = 500
data_buffer = {"x": [], "y": [], "z": []}

def plot_gaze_data():
    """用 Matplotlib 繪製 gaze XY 與 Z 軌跡圖，並轉成 NumPy 圖片"""
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 左右併排

    # 左下角 - XY 軌跡圖
    axs[0].set_title("Gaze XY Trajectory")
    axs[0].set_xlim(-1000, 1000)
    axs[0].set_ylim(-1000, 1000)
    axs[0].set_xlabel("Gaze X (mm)")
    axs[0].set_ylabel("Gaze Y (mm)")
    axs[0].grid()
    axs[0].plot(data_buffer["x"], data_buffer["y"], 'bo-', markersize=3, alpha=0.5)

    # 右下角 - Z 軸波形圖
    axs[1].set_xlim(0, buffer_size)
    axs[1].set_ylim(0, 1500)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Gaze Z (mm)")
    axs[1].set_title("Gaze Z Waveform")
    axs[1].grid()
    axs[1].plot(range(len(data_buffer["z"])), data_buffer["z"], 'b-')

    fig.canvas.draw()

    # 轉換 Matplotlib 圖片為 NumPy 陣列
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # 關閉 Matplotlib 圖片，釋放記憶體
    return img

def main():
    address, port = get_ip_and_port()
    sc = SyncClient(address, port)
    if not sc.get_status().eye_image_encoding_enabled:
        print('Warning: Please enable eye image encoding and try again.')
        return

    th = sc.create_streaming_thread(StreamingMode.GAZE_SCENE_EYES)
    th.start()

    try:
        while True:
            frame_data = sc.get_scene_frames_from_streaming(timeout=5.0)
            frame_datum = frame_data[-1]  # get the last frame
            buffer = frame_datum.get_buffer()
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            gaze = find_nearest_timestamp_match(frame_datum.get_timestamp(), gazes)
            left_eye_data = sc.get_left_eye_frames_from_streaming(timeout=5.0)
            left_eye_datum = find_nearest_timestamp_match(frame_datum.get_timestamp(), left_eye_data)
            right_eye_data = sc.get_right_eye_frames_from_streaming(timeout=5.0)
            right_eye_datum = find_nearest_timestamp_match(frame_datum.get_timestamp(), right_eye_data)

            # 取得 gaze 數據
            data_buffer["x"].append(gaze.combined.gaze_2d.x)
            data_buffer["y"].append(gaze.combined.gaze_2d.y)
            data_buffer["z"].append(gaze.combined.gaze_3d.z)

            # 移除舊數據，維持固定長度
            if len(data_buffer["x"]) > buffer_size:
                data_buffer["x"].pop(0)
                data_buffer["y"].pop(0)
                data_buffer["z"].pop(0)

            # Overlay gaze on scene camera frame
            center = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))
            radius = 30
            bgr_color = (255, 255, 0)
            thickness = 5
            cv2.circle(buffer, center, radius, bgr_color, thickness)

            resize_ratio = 0.5
            # Overlay left eye on scene camera frame
            left_eye_frame = left_eye_datum.get_buffer()
            draw_to_center_top(buffer, left_eye_frame, Camera.LEFT_EYE, resize_ratio)

            # Overlay right eye on scene camera frame
            right_eye_frame = right_eye_datum.get_buffer()
            draw_to_center_top(buffer, right_eye_frame, Camera.RIGHT_EYE, resize_ratio)

            # 繪製 gaze XY 與 Z 軌跡圖
            gaze_plot = plot_gaze_data()
            gaze_plot = cv2.resize(gaze_plot, (400, 200))  # 調整大小
            gaze_plot = cv2.cvtColor(gaze_plot, cv2.COLOR_RGB2BGR)  # Matplotlib 預設是 RGB，要轉 BGR

            # 取得影像尺寸
            h, w, _ = buffer.shape
            ph, pw, _ = gaze_plot.shape

            # 左下角（XY 軌跡圖）
            buffer[h - ph:h, 0:pw] = gaze_plot[:, :pw // 2]

            # 右下角（Z 軌跡波形圖）
            buffer[h - ph:h, w - pw // 2:w] = gaze_plot[:, pw // 2:]

            cv2.imshow('Press "q" to exit', buffer)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as ex:
        print(ex)
    finally:
        th.cancel()
        th.join()
        print('Stopped')

def draw_to_center_top(scene_cam_frame: np.ndarray, 
                       eye_frame: np.ndarray, camera: Camera, 
                       ratio: float = 1.0, center_margin=5):
    half_frame_width = scene_cam_frame.shape[1] // 2
    pos_x = half_frame_width
    pos_y = 0
    resized_eye_width = int(eye_frame.shape[1] * ratio)
    resized_eye_height = int(eye_frame.shape[0] * ratio)
    resized_eye = cv2.resize(eye_frame, (resized_eye_width, resized_eye_height))

    if camera == Camera.LEFT_EYE:
        pos_x -= (resized_eye_width + center_margin)
    elif camera == Camera.RIGHT_EYE:
        pos_x += center_margin
    else:
        raise ValueError(f"Invalid camera type: {camera}")

    scene_cam_frame[pos_y:pos_y + resized_eye_height, 
                    pos_x:pos_x + resized_eye_width] = resized_eye

if __name__ == '__main__':
    main()
