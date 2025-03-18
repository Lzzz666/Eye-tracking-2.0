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
from pyzbar.pyzbar import decode
from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient

# ------------- 初始化 -------------
data_buffer = {"x": [], "y": [], "z": []}
buffer_size = 500  

address, port = get_ip_and_port()
sc = SyncClient(address, port)
th = sc.create_streaming_thread(StreamingMode.GAZE_SCENE_EYES)  # 啟用眼睛 & 場景影像
th.start()

# 影像初始化
left_eye_img = np.zeros((100, 100, 3), dtype=np.uint8)  # 黑色影像
right_eye_img = np.zeros((100, 100, 3), dtype=np.uint8)
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
    將左眼和右眼影像垂直排列，並放置於場景影像的左側
    :param left_eye: 左眼影像
    :param right_eye: 右眼影像
    :param scene_img: 場景影像
    :return: 合併後的影像
    """
    # 調整左右眼影像大小，保持比例
    eye_height = max(left_eye.shape[0], right_eye.shape[0])
    left_eye_resized = cv2.resize(left_eye, (int(left_eye.shape[1] * eye_height / left_eye.shape[0]), eye_height))
    right_eye_resized = cv2.resize(right_eye, (int(right_eye.shape[1] * eye_height / right_eye.shape[0]), eye_height))

    # 眼部影像垂直合併
    eye_stack = np.vstack((left_eye_resized, right_eye_resized))

    # 確保與場景影像的高度一致
    scene_height = scene_img.shape[0]
    if eye_stack.shape[0] < scene_height:
        top_pad = (scene_height - eye_stack.shape[0]) // 2
        bottom_pad = scene_height - eye_stack.shape[0] - top_pad
        final_eye_stack = cv2.copyMakeBorder(eye_stack, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        final_eye_stack = cv2.resize(eye_stack, (eye_stack.shape[1], scene_height))  # 只有超過時才縮小

    stacked_image = np.hstack((final_eye_stack, scene_img))

    return stacked_image

# 模擬按鈕功能
def mouse_callback(event, x, y, flags, param):
    global recording, video_writer, final_image, csv_file, csv_writer, output_dir  # ✅ 加入 csv_writer

    button_x, button_y, button_w, button_h = 0, 0, 150, 50  

    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            recording = not recording  
            print(f"Recording started: {recording}")

            if recording:
                output_dir = create_output_dir()
                cv2.putText(final_image, 'Recording...', (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                video_writer = cv2.VideoWriter(f'{output_dir}/recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (final_image.shape[1], final_image.shape[0]))

                csv_filename = os.path.join(output_dir, "data.csv")
                csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')  
                csv_writer = csv.writer(csv_file)  # ✅ 初始化 csv_writer
                csv_writer.writerow(["Timestamp", "Gaze X", "Gaze Y", "Gaze Z"])
                print(f"🔴 開始錄製: {output_dir}")

            else:
                cv2.putText(final_image, 'Recording Stopped', (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                video_writer.release()
                
                if csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None  # ✅ 清除 csv_writer，避免錯誤
                    print(f"🛑 錄製結束，檔案存放於: {output_dir}")

# 設置滑鼠事件回調函數
cv2.setMouseCallback("Gaze Tracking View", mouse_callback)

# 畫出按鈕
def draw_button(frame):
    button_x, button_y, button_w, button_h = 0, 0, 150, 50  # 按鈕的位置和大小
    button_color = (0, 255, 0) if recording else (0, 0, 255)  # 根據錄製狀態設定顏色
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)  # 繪製按鈕背景
    cv2.putText(frame, "Start/Stop Recording", (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 按鈕文本

# ------------- QR Code 掃描 -------------
def scan_qrcode(image: np.ndarray):
    """ 嘗試掃描影像中的 QR Code，並返回內容 """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉換為灰階
    detector = cv2.QRCodeDetector()  # OpenCV 內建 QR Code 掃描器
    data, bbox, _ = detector.detectAndDecode(gray)

    if data:  # 如果偵測到 QR Code
        return data

    # 使用 pyzbar 嘗試解碼
    decoded_objects = decode(gray)
    for obj in decoded_objects:
        return obj.data.decode("utf-8")  # 解析 QR Code 內容

    return None  # 沒有偵測到 QR Code
# ------------- 更新數據 -------------
def update_data():
    global data_buffer, left_eye_img, right_eye_img, scene_img, final_image, video_writer, csv_writer, csv_file 
    try:
        while True:
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            scene_frames = sc.get_scene_frames_from_streaming(timeout=5.0)
            left_eye_frames = sc.get_left_eye_frames_from_streaming(timeout=5.0)
            right_eye_frames = sc.get_right_eye_frames_from_streaming(timeout=5.0)

            # 更新 gaze 數據
            for gaze in gazes:
                if gaze.combined.gaze_3d.validity:
                    x, y, z = gaze.combined.gaze_3d.x, gaze.combined.gaze_3d.y, gaze.combined.gaze_3d.z
                    timestamp = time.time()

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
                            csv_writer.writerow([timestamp, x, y, z])  # ✅ 寫入時間戳與 gaze 數據
                            csv_file.flush()  # 🔥 立即寫入文件，防止意外丟失數據

            # 更新影像
            if scene_frames:
                scene_img = scene_frames[-1].get_buffer()

            # 擷取影像中心部分（假設 QR Code 在畫面中央）
            h, w, _ = scene_img.shape
            center_crop = scene_img[h//4:3*h//4, w//4:3*w//4]  # 取畫面中間 50% 區域

            # 嘗試掃描 QR Code
            qr_result = scan_qrcode(center_crop)
            if qr_result:
                print(f"📷 QR Code Detected: {qr_result}")

            if left_eye_frames:
                left_eye_img = left_eye_frames[-1].get_buffer()
            if right_eye_frames:
                right_eye_img = right_eye_frames[-1].get_buffer()

            # 合併影像（眼睛影像放左側）
            final_image = stack_eye_images(left_eye_img, right_eye_img, scene_img)

            # 畫出按鈕
            draw_button(final_image)

            # 顯示影像
            if recording and video_writer:
                video_writer.write(final_image)  # 錄製影像

            cv2.imshow("Gaze Tracking View", final_image)

            # 按下 Q 結束
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            time.sleep(0.05)
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

