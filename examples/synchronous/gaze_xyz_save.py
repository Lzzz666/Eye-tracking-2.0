import csv
import os
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient
from ganzin.sol_sdk.utils import find_nearest_timestamp_match

# ------------- 初始化 -------------
data_buffer = {"x": [], "y": [], "z": []}
buffer_size = 500  

address, port = get_ip_and_port()
sc = SyncClient(address, port)
th = sc.create_streaming_thread(StreamingMode.GAZE_SCENE_EYES)  # 啟用眼睛 & 場景影像
th.start()

# 初始化錄製狀態
recording = False
output_dir = None
csv_writer = None

# 調整布局
left_eye_img = np.zeros((100, 100, 3), dtype=np.uint8)  # 預設黑色影像
right_eye_img = np.zeros((100, 100, 3), dtype=np.uint8)
scene_img = np.zeros((480, 640, 3), dtype=np.uint8)

def update_data():
    global data_buffer, left_eye_img, right_eye_img, scene_img
    try:
        while True:
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            scene_frames = sc.get_scene_frames_from_streaming(timeout=5.0)
            left_eye_frames = sc.get_left_eye_frames_from_streaming(timeout=5.0)
            right_eye_frames = sc.get_right_eye_frames_from_streaming(timeout=5.0)

            for gaze in gazes:
                if gaze.combined.gaze_3d.validity:
                    x, y, z = gaze.combined.gaze_3d.x, gaze.combined.gaze_3d.y, gaze.combined.gaze_3d.z
                    if len(data_buffer["x"]) >= buffer_size:
                        data_buffer["x"].pop(0)
                        data_buffer["y"].pop(0)
                        data_buffer["z"].pop(0)
                    data_buffer["x"].append(x)
                    data_buffer["y"].append(y)
                    data_buffer["z"].append(z)

            # 更新影像 (確保時間同步)
            if scene_frames:
                scene_img = scene_frames[-1].get_buffer()

            if left_eye_frames:
                left_eye_img = left_eye_frames[-1].get_buffer()

            if right_eye_frames:
                right_eye_img = right_eye_frames[-1].get_buffer()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("🔴 測試已中斷")
    finally:
        print("🛑 停止串流")
        th.cancel()
        th.join()

threading.Thread(target=update_data, daemon=True).start()

# ------------- 設定 Matplotlib -------------
# 建立圖表
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Real-time Gaze Tracking")

# 定義 GridSpec
gs = GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

# 左上: 左眼影像
ax_left = fig.add_subplot(gs[0, 0])
ax_left.set_title("Left Eye")
eye_left_img = ax_left.imshow(left_eye_img, aspect="auto")
ax_left.axis("off")

# 左中: 右眼影像
ax_right = fig.add_subplot(gs[1, 0])
ax_right.set_title("Right Eye")
eye_right_img = ax_right.imshow(right_eye_img, aspect="auto")
ax_right.axis("off")

# 右側: 場景影像 (佔據兩列)
scene_ax = fig.add_subplot(gs[:2, 1])
scene_ax.set_title("Scene View")
scene_img_artist = scene_ax.imshow(scene_img, aspect="auto")  # 儲存 imshow 的返回值
scene_ax.axis("off")

# 左下: XY 軌跡圖
ax_xy = fig.add_subplot(gs[2, 0])
ax_xy.set_title("Gaze XY Trajectory")
ax_xy.set_xlim(-1000, 1000)
ax_xy.set_ylim(-1000, 1000)
ax_xy.set_xlabel("Gaze X (mm)")
ax_xy.set_ylabel("Gaze Y (mm)")
ax_xy.grid()
xy_line, = ax_xy.plot([], [], 'bo-', markersize=3, alpha=0.5)

# 右下: Z 軸波形圖
ax_z = fig.add_subplot(gs[2, 1])
ax_z.set_xlim(0, buffer_size)
ax_z.set_ylim(0, 1500)
ax_z.set_ylabel("Gaze Z (mm)")
ax_z.set_title("Gaze Z Waveform")
ax_z.grid()
z_line, = ax_z.plot([], [], 'b-', label="Gaze Z")
ax_z.set_xlabel("Time Steps")

def update_plot(frame):
    # 更新 XY 與 Z 軌跡
    xy_line.set_data(data_buffer["x"], data_buffer["y"])
    z_line.set_data(range(len(data_buffer["z"])), data_buffer["z"])

    # 更新影像
    eye_left_img.set_data(left_eye_img)
    eye_right_img.set_data(right_eye_img)
    scene_img_artist.set_data(scene_img)  # 使用 scene_img_artist 而不是 scene_ax

    # 如果正在錄製，則保存影像和數據
    if recording:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 保存影像
        mpimg.imsave(os.path.join(output_dir, "left_eye", f"{timestamp}.png"), left_eye_img)
        mpimg.imsave(os.path.join(output_dir, "right_eye", f"{timestamp}.png"), right_eye_img)
        mpimg.imsave(os.path.join(output_dir, "scene", f"{timestamp}.png"), scene_img)

        # 保存數據
        with open(os.path.join(output_dir, "gaze_data.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + data_buffer["x"][-1:] + data_buffer["y"][-1:] + data_buffer["z"][-1:])
    
    return [xy_line, z_line, eye_left_img, eye_right_img, scene_img_artist]

def on_key(event):
    global recording, output_dir, csv_writer
    if event.key == 'r':  # 開始錄製
        recording = True
        # 創建資料夾，並準備 CSV 檔案
        output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        left_eye_dir = os.path.join(output_dir, "left_eye")
        right_eye_dir = os.path.join(output_dir, "right_eye")
        scene_dir = os.path.join(output_dir, "scene")
        
        os.makedirs(left_eye_dir, exist_ok=True)
        os.makedirs(right_eye_dir, exist_ok=True)
        os.makedirs(scene_dir, exist_ok=True)

        with open(os.path.join(output_dir, "gaze_data.csv"), mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Timestamp", "Gaze X", "Gaze Y", "Gaze Z"])  # CSV 標題
        print(f"錄製開始，資料夾：{output_dir}")

    elif event.key == 's':  # 停止錄製
        recording = False
        print("錄製停止")

fig.canvas.mpl_connect('key_press_event', on_key)

ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)
try:
    plt.show(block=False)
    plt.show()
except KeyboardInterrupt:
    print("User interrupted. Exiting...")
    plt.close('all')  # 關閉所有圖形
