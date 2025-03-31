import socket
import json
import sys
import threading
import time

# UDP configuration
UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 5100     # Port to listen on

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Global variable to store received Joy-Con data
# Global variables to store the latest sensor data
latest_data = {
    'accel': [0, 0, 0],
    'gyro': [0, 0, 0],
    'accel1': [0, 0, 0],
    'gyro1': [0, 0, 0]
}
lock = threading.Lock()  # Protect joycon_data during concurrent access

# UDP configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5000

# Global variables
display = None
orientation_right = [0, 0, 0]  # roll, pitch, yaw for right Joy-Con
orientation_left = [0, 0, 0]   # roll, pitch, yaw for left Joy-Con

def data_fetcher():
    """Thread that continuously fetches data from the UDP server"""
    global latest_data, orientation_right, orientation_left
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    # print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

    while True:
        data, addr = sock.recvfrom(4096)
        try:
            data = json.loads(data.decode('utf-8'))
            
            # Update global variables
            with threading.Lock():
                # Right Joy-Con data
                if 'accel' in data and isinstance(data['accel'], dict):
                    latest_data['accel'] = [
                        data['accel']['x'],
                        data['accel']['y'],
                        data['accel']['z']
                    ]
                
                if 'gyro' in data and isinstance(data['gyro'], dict):
                    latest_data['gyro'] = [
                        data['gyro']['x'],
                        data['gyro']['y'],
                        data['gyro']['z']
                    ]
                    # Use the default noise indicator to make a deadzone for the gyro data
                    if abs(latest_data['gyro'][0]) > 75:
                        orientation_right[0] = latest_data['gyro'][0] * 0.01  # Roll
                    if abs(latest_data['gyro'][2]) > 75:
                        orientation_right[1] = latest_data['gyro'][2] * 0.01  # Pitch
                    if abs(latest_data['gyro'][1]) > 75:
                        orientation_right[2] = latest_data['gyro'][1] * 0.01  # Yaw
                    
                
                # Left Joy-Con data
                if 'accel1' in data and isinstance(data['accel1'], dict):
                    latest_data['accel1'] = [
                        data['accel1']['x'],
                        data['accel1']['y'],
                        data['accel1']['z']
                    ]
                
                if 'gyro1' in data and isinstance(data['gyro1'], dict):
                    latest_data['gyro1'] = [
                        data['gyro1']['x'],
                        data['gyro1']['y'],
                        data['gyro1']['z']
                    ]
                    # Use the default noise indicator to make a deadzone for the gyro data
                    if abs(latest_data['gyro1'][0]) > 75:
                        orientation_left[0] = latest_data['gyro1'][0] * 0.01  # Roll
                    if abs(latest_data['gyro1'][2]) > 75:
                        orientation_left[1] = latest_data['gyro1'][2] * 0.01  # Pitch
                    if abs(latest_data['gyro1'][1]) > 75:
                        orientation_left[2] = latest_data['gyro1'][1] * 0.01  # Yaw
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"Error processing data: {e}")


def data_producer():
    global latest_data
    while True:
        with lock:
            if latest_data:
                # Extract relevant data
                # data = {
                #     "accel": latest_data['accel'],
                #     "gyro": latest_data['gyro'],
                #     "accel1": latest_data['accel1'],
                #     "gyro1": latest_data['gyro1']
                # }
                
                data = {
                    "accel": latest_data['accel'],
                    "gyro": orientation_right,
                    "accel1": latest_data['accel1'],
                    "gyro1": orientation_left
                }
                print(json.dumps(data))  # Print the data for consumption by Isaac Sim
                sys.stdout.flush()
                
        time.sleep(0.03)  # Rate at which data is produced

if __name__ == '__main__':
    # Start the receiver thread
    receiver_thread = threading.Thread(target=data_fetcher)
    receiver_thread.daemon = True  # Exit if the main thread exits
    receiver_thread.start()
    
    # Start the data producer thread
    producer_thread = threading.Thread(target=data_producer)
    producer_thread.daemon = True  # Exit if the main thread exits
    producer_thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")
        sys.exit(0)
