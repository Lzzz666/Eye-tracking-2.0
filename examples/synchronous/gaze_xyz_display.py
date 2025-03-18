import sys
import os

# Add the parent directory of 'synchronous' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient

def main():
    address, port = get_ip_and_port()
    sc = SyncClient(address, port)

    th = sc.create_streaming_thread(StreamingMode.GAZE)
    th.start()

    try:
        while True:
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            for gaze in gazes:
                # print(f'gaze: x = {gaze.combined.gaze_2d.x}, y = {gaze.combined.gaze_2d.y}')
                print(f'origin: left_eye: x = {gaze.left_eye.gaze.origin.x}, y = {gaze.left_eye.gaze.origin.y},z = {gaze.left_eye.gaze.origin.z} | right_eye: x = {gaze.right_eye.gaze.origin.x}, y = {gaze.right_eye.gaze.origin.y}, z = {gaze.right_eye.gaze.origin.z}')
                print(f'direction: left_eye: x = {gaze.left_eye.gaze.direction.x}, y = {gaze.left_eye.gaze.direction.y},z = {gaze.left_eye.gaze.direction.z} | right_eye: x = {gaze.right_eye.gaze.direction.x}, y = {gaze.right_eye.gaze.direction.y}, z = {gaze.right_eye.gaze.direction.z}')
    except KeyboardInterrupt: # Press Ctrl-C to stop
        pass
    except Exception as ex:
        print(ex)
    finally:
        print('Stopped')

    th.cancel()
    th.join()

if __name__ == '__main__':
    main()
