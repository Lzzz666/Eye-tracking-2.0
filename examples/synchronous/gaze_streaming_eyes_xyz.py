import sys
import os
import time
import csv
import threading
import numpy as np
import cv2  # OpenCV
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
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

# 影像初始化
left_eye_img = np.zeros((400, 400, 3), dtype=np.uint8)  # 黑色影像
right_eye_img = np.zeros((400, 400, 3), dtype=np.uint8)
scene_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Gaze Tracking View", 0)  # 允許調整視窗大小
cv2.resizeWindow("Gaze Tracking View", 1200, 900)  # 設置視窗為1200x900
final_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 預設一個空的影像

recording = False  # 標記是否正在錄製
video_writer = None  # 用來錄製影片的 VideoWriter
csv_file = None
csv_writer = None
output_dir = None


def create_output_dir():
    """建立以當前時間為名稱的資料夾"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")  
    directory = f"records/{timestamp}"
    os.makedirs(directory, exist_ok=True)
    return directory


def stack_eye_images(left_eye: np.ndarray, right_eye: np.ndarray, scene_img: np.ndarray) -> np.ndarray:
    """
    將左眼和右眼影像垂直排列，並放置於場景影像的左側，保持原始大小。
    :param left_eye: 左眼影像
    :param right_eye: 右眼影像
    :param scene_img: 場景影像
    :return: 合併後的影像
    """
    # 確保左右眼影像垂直排列
    eye_height = left_eye.shape[0] + right_eye.shape[0]  # 左右眼高度總和
    left_eye_resized = left_eye
    right_eye_resized = right_eye

    # 眼部影像垂直合併
    eye_stack = np.vstack((left_eye_resized, right_eye_resized))

    # 確保與場景影像的高度一致
    scene_height = scene_img.shape[0]
    if eye_stack.shape[0] < scene_height:
        top_pad = (scene_height - eye_stack.shape[0]) // 2
        bottom_pad = scene_height - eye_stack.shape[0] - top_pad
        final_eye_stack = cv2.copyMakeBorder(eye_stack, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        final_eye_stack = eye_stack  # 直接使用合併後的影像，避免壓縮

    stacked_image = np.hstack((final_eye_stack, scene_img))

    return stacked_image
# 模擬按鈕功能
def mouse_callback(event, x, y, flags, param):
    global recording, video_writer, final_image, csv_file, csv_writer, output_dir  # ✅ 加入 csv_writer

    button_x, button_y, button_w, button_h = 10, 10, 80, 40 

    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            recording = not recording  
            print(f"Recording started: {recording}")

            if recording:
                output_dir = create_output_dir()
                cv2.putText(final_image, "Rec...", (button_x + 20, button_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
                
                video_writer = cv2.VideoWriter(f'{output_dir}/recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 19, (final_image.shape[1], final_image.shape[0]))

                csv_filename = os.path.join(output_dir, "data.csv")
                csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')  
                csv_writer = csv.writer(csv_file)  # ✅ 初始化 csv_writer
                csv_writer.writerow(["Timestamp", "Gaze X", "Gaze Y", "Gaze Z","Left Eye X", "Left Eye Y", "Left Eye Z", "Right Eye X", "Right Eye Y", "Right Eye Z", "Left Eye Dir X", "Left Eye Dir Y", "Left Eye Dir Z", "Right Eye Dir X", "Right Eye Dir Y", "Right Eye Dir Z"])  # ✅ 寫入標題
                print(f"🔴 開始錄製: {output_dir}")

            else:
                cv2.putText(final_image, 'Recording Stopped', (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                video_writer.release()
                
                if csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None  # ✅ 清除 csv_writer，避免錯誤
                    print(f"🛑 錄製結束，檔案存放於: {output_dir}")

cv2.setMouseCallback("Gaze Tracking View", mouse_callback)

# 畫出按鈕
def draw_button(frame,timestamp):
    button_x, button_y, button_w, button_h = 10, 10, 120, 40  # 縮小按鈕
    button_color = (0, 255, 0) if recording else (0, 0, 255)  # 錄製狀態變色
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)  # 按鈕背景

    cv2.putText(frame, "Rec", (button_x + 20, button_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  
    timestamp_text = f"{timestamp:.1f}"
    cv2.putText(frame, timestamp_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ------------- 更新數據 -------------
def update_data():
    global data_buffer, left_eye_img, right_eye_img, scene_img, final_image, video_writer, csv_writer, csv_file 
    frame_count = 0  # 記錄處理的幀數
    start_time = time.time()  # 開始時間

    try:
        while True:
            # test frame cnt
            frame_count += 1
            # 每一秒更新一次幀率
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # 每秒更新一次幀率
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")  # 顯示當前幀率
                
                # 重設時間計數
                start_time = time.time()
                frame_count = 0
    
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            scene_frames = sc.get_scene_frames_from_streaming(timeout=5.0)
            frame_datum = scene_frames[-1] 

            left_eye_frames = sc.get_left_eye_frames_from_streaming(timeout=5.0)
            right_eye_frames = sc.get_right_eye_frames_from_streaming(timeout=5.0)

            # 更新 gaze 數據
            gaze = find_nearest_timestamp_match(frame_datum.get_timestamp(), gazes)
            x, y, z = gaze.combined.gaze_3d.x, gaze.combined.gaze_3d.y, gaze.combined.gaze_3d.z
            x_left, y_left, z_left = gaze.left_eye.gaze.origin.x, gaze.left_eye.gaze.origin.y, gaze.left_eye.gaze.origin.z
            x_right, y_right, z_right = gaze.right_eye.gaze.origin.x, gaze.right_eye.gaze.origin.y, gaze.right_eye.gaze.origin.z
            x_left_dir, y_left_dir, z_left_dir = gaze.left_eye.gaze.direction.x, gaze.left_eye.gaze.direction.y, gaze.left_eye.gaze.direction.z
            x_right_dir, y_right_dir, z_right_dir = gaze.right_eye.gaze.direction.x, gaze.right_eye.gaze.direction.y, gaze.right_eye.gaze.direction.z
            timestamp = frame_datum.get_timestamp()

            if len(data_buffer["x"]) >= buffer_size:
                data_buffer["x"].pop(0)
                data_buffer["y"].pop(0)
                data_buffer["z"].pop(0)
            data_buffer["x"].append(x)
            data_buffer["y"].append(y)
            data_buffer["z"].append(z)

            # 🚀 新增寫入 CSV
            if recording:
                if csv_writer is None:  # 🔥 確保 csv_writer 不為 None
                    print("⚠️ 錯誤: csv_writer 尚未初始化！")
                else:
                    csv_writer.writerow([timestamp, x, y, z, x_left, y_left, z_left, x_right, y_right, z_right, x_left_dir, y_left_dir, z_left_dir, x_right_dir, y_right_dir, z_right_dir])  # ✅ 寫入時間戳與 gaze 數據
                    csv_file.flush()  # 🔥 立即寫入文件，防止意外丟失數據

            # 更新影像
            if scene_frames:
                scene_img = scene_frames[-1].get_buffer()
            if left_eye_frames:
                left_eye_img = left_eye_frames[-1].get_buffer()
            if right_eye_frames:
                right_eye_img = right_eye_frames[-1].get_buffer()

            # 合併影像（眼睛影像放左側）
            final_image = stack_eye_images(left_eye_img, right_eye_img, scene_img)
                
            # 畫出按鈕
            draw_button(final_image,timestamp)

            # 顯示影像
            if recording and video_writer:
                video_writer.write(final_image)  # 錄製影像

            cv2.imshow("Gaze Tracking View", final_image)

            # 按下 Q 結束
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("🔴 測試已中斷")
    finally:
        print("🛑 停止串流")
        th.cancel()
        th.join()
        cv2.destroyAllWindows()


threading.Thread(target=update_data, daemon=True).start()

# ------------- 設定 Matplotlib -------------
fig = plt.figure(figsize=(8, 6))
fig.suptitle("Real-time Gaze Tracking")

gs = GridSpec(2, 1, height_ratios=[2, 1])

# XY 軌跡圖
ax_xy = fig.add_subplot(gs[0, 0])
ax_xy.set_title("Gaze XY Trajectory")
ax_xy.set_xlim(-1000, 1000)
ax_xy.set_ylim(-1000, 1000)
ax_xy.set_xlabel("Gaze X (mm)")
ax_xy.set_ylabel("Gaze Y (mm)")
ax_xy.grid()
xy_line, = ax_xy.plot([], [], 'bo-', markersize=3, alpha=0.5)

# Z 軸波形圖
ax_z = fig.add_subplot(gs[1, 0])
ax_z.set_xlim(0, buffer_size)
ax_z.set_ylim(0, 1500)
ax_z.set_ylabel("Gaze Z (mm)")
ax_z.set_title("Gaze Z Waveform")
ax_z.grid()
z_line, = ax_z.plot([], [], 'b-', label="Gaze Z")
ax_z.set_xlabel("Time Steps")

def update_plot(frame):
    xy_line.set_data(data_buffer["x"], data_buffer["y"])
    z_line.set_data(range(len(data_buffer["z"])), data_buffer["z"])
    return [xy_line, z_line]
ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False) 

try:
    plt.show(block=False)
    plt.show()
except KeyboardInterrupt:
    print("User interrupted. Exiting...")
    plt.close('all')

