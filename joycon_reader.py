# # from flask import Flask, request
# # import json
# # import time
# # import threading
# # import sys

# # app = Flask(__name__)
# # joycon_data = None  # Global variable to store received Joy-Con data
# # lock = threading.Lock() # Protect joycon_data during concurrent access

# # @app.route('/update', methods=['POST'])
# # def update_joycon_data():
# #     global joycon_data
# #     with lock:
# #         joycon_data = request.get_json()
# #     return "Data received", 200

# # def data_producer():
# #     global joycon_data
# #     while True:
# #         with lock:
# #             if joycon_data:
# #                 # Extract relevant data
# #                 data = {
# #                     "accel": joycon_data.get('accel'),
# #                     "gyro": joycon_data.get('gyro'),
# #                     "buttons": joycon_data.get('buttons'),
# #                     "analog-sticks": joycon_data.get('analog-sticks'),
# #                     "battery": joycon_data.get('battery'),
# #                     "accel1": joycon_data.get('accel1'),
# #                     "gyro1": joycon_data.get('gyro1'),
# #                     "buttons1": joycon_data.get('buttons1'),
# #                     "analog-sticks1": joycon_data.get('analog-sticks1'),
# #                     "battery1": joycon_data.get('battery1')
# #                 }
# #                 print(json.dumps(data)) # Print the data for consumption by Isaac Sim
# #                 # flush
# #                 sys.stdout.flush()
# #                 # also print to stderr
# #                 sys.stderr.write(json.dumps(data) + '\n')
# #                 sys.stderr.flush()
# #         time.sleep(0.03)  # Rate at which data is produced

# # if __name__ == '__main__':
# #     # Start the data production in a separate thread
# #     producer_thread = threading.Thread(target=data_producer)
# #     producer_thread.daemon = True # Exit if the main thread exits
# #     producer_thread.start()

# #     app.run(debug=False, host='0.0.0.0', port=5000) # Start Flask server

# import asyncio
# import json
# import sys
# import websockets

# joycon_data = None  # Global variable to store received Joy-Con data
# lock = asyncio.Lock()  # Protect joycon_data during concurrent access

# async def update_joycon_data(websocket):
#     global joycon_data
#     async for message in websocket:
#         data = json.loads(message)
#         async with lock:
#             joycon_data = data
#         await websocket.send("Data received")

# async def data_producer():
#     global joycon_data
#     while True:
#         await asyncio.sleep(0.03)  # Rate at which data is produced
#         async with lock:
#             if joycon_data:
#                 data = {
#                     "accel": joycon_data.get('accel'),
#                     "gyro": joycon_data.get('gyro'),
#                     "buttons": joycon_data.get('buttons'),
#                     "analog-sticks": joycon_data.get('analog-sticks'),
#                     "battery": joycon_data.get('battery'),
#                     "accel1": joycon_data.get('accel1'),
#                     "gyro1": joycon_data.get('gyro1'),
#                     "buttons1": joycon_data.get('buttons1'),
#                     "analog-sticks1": joycon_data.get('analog-sticks1'),
#                     "battery1": joycon_data.get('battery1')
#                 }
#                 print(json.dumps(data))  # Print the data for consumption by Isaac Sim
#                 sys.stdout.flush()
#     print("Data producer stopped")

# async def main():
#     server = await websockets.serve(update_joycon_data, "0.0.0.0", 5000)
#     producer = asyncio.create_task(data_producer())
#     await asyncio.gather(producer, server.wait_closed())

# if __name__ == '__main__':
#     asyncio.run(main())


import socket
import json
import sys
import threading
import time

# UDP configuration
UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 5000     # Port to listen on

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Global variable to store received Joy-Con data
joycon_data = None
lock = threading.Lock()  # Protect joycon_data during concurrent access

def receive_data():
    global joycon_data
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")
    
    while True:
        # Receive data from socket
        data, addr = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        
        try:
            # Decode and parse JSON
            json_data = json.loads(data.decode('utf-8'))
            
            # Store the data with thread safety
            with lock:
                joycon_data = json_data
                
        except json.JSONDecodeError:
            print("Error: Received invalid JSON data")
        except Exception as e:
            print(f"Error processing received data: {str(e)}")

def data_producer():
    global joycon_data
    while True:
        with lock:
            if joycon_data:
                # Extract relevant data
                data = {
                    "accel": joycon_data.get('accel'),
                    "gyro": joycon_data.get('gyro'),
                    "buttons": joycon_data.get('buttons'),
                    "analog-sticks": joycon_data.get('analog-sticks'),
                    "battery": joycon_data.get('battery'),
                    "accel1": joycon_data.get('accel1'),
                    "gyro1": joycon_data.get('gyro1'),
                    "buttons1": joycon_data.get('buttons1'),
                    "analog-sticks1": joycon_data.get('analog-sticks1'),
                    "battery1": joycon_data.get('battery1')
                }
                print(json.dumps(data))  # Print the data for consumption by Isaac Sim
                # sys.stdout.flush()
                
        time.sleep(0.03)  # Rate at which data is produced

if __name__ == '__main__':
    # Start the receiver thread
    receiver_thread = threading.Thread(target=receive_data)
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
