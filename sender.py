import requests
import time
from pyjoycon import JoyCon, get_R_id, get_L_id
joycon_id = get_R_id()
joycon = JoyCon(*joycon_id)
print(f"Connected to Joy-Con (Right) with ID: {joycon_id}")
joycon_id1 = get_L_id()
joycon1 = JoyCon(*joycon_id1)
print(f"Connected to Joy-Con (Left) with ID: {joycon_id1}")

while True:
    status = joycon.get_status()
    status1 = joycon1.get_status()
    combined_status = {**status, **{'accel1': status1['accel'], 'gyro1': status1['gyro'], 'buttons1': status1['buttons'], 'analog-sticks1': status1['analog-sticks'], 'battery1': status1['battery']}}
    requests.post(f'http://100.119.136.161:5000/update', json=combined_status)
    time.sleep(0.03)


# from pyjoycon import GyroTrackingJoyCon, get_R_id
# import time

# joycon_id = get_R_id()
# joycon = GyroTrackingJoyCon(*joycon_id)
# while True:
#     print("joycon pointer:  ", joycon.pointer)
#     print("joycon rotation: ", joycon.rotation)
#     print("joycon direction:", joycon.direction)
#     print()
#     time.sleep(0.05)


# add a visualization for this in 3d for both R and L controller


import socket
import json
import time
from pyjoycon import GyroTrackingJoyCon, get_R_id, get_L_id
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
joycon_r = GyroTrackingJoyCon(*get_R_id())
joycon_l = GyroTrackingJoyCon(*get_L_id())
joycon_r.calibrate(seconds=5)
joycon_l.calibrate(seconds=5)
while True:
    status = joycon_r.get_status()
    status1 = joycon_l.get_status()
    data = {**status, **{'accel1': status1['accel'], 'gyro1': status1['gyro'], 'buttons1': status1['buttons'], 'analog-sticks1': status1['analog-sticks'], 'battery1': status1['battery']}}
    message = json.dumps(data).encode('utf-8')
    print(message)
    sock.sendto(message, ("127.0.0.1", 5000))  # Change IP to your target
    time.sleep(0.03)
