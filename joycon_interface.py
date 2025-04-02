import numpy as np
import threading
import subprocess
import json
import torch
import time
import sys

class BimanualTeleopInterface:
    """Base class for bimanual teleoperation interfaces that track position and rotation of two controllers"""
    
    def __init__(self):
        # Initialize positions and rotations for both controllers
        self.left_position = np.array([0.0, 0.0, 0.0])  # Position of first controller
        self.right_position = np.array([0.0, 1.0, 0.0])  # Position of second controller (offset to distinguish)
        
        self.left_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Rotation of first controller (quaternion)
        self.right_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Rotation of second controller (quaternion)
        
        # Initialize velocity vectors 
        self.velocity1 = np.array([0.0, 0.0, 0.0])
        self.velocity2 = np.array([0.0, 0.0, 0.0])

    # safe getter
    def get_lr_pos_rot_safe(self):
        """Get positions and rotations of both controllers in a thread-safe manner"""
        with self.data_lock:
            left_pos = self.left_position.copy()
            right_pos = self.right_position.copy()
            left_rot = self.left_rotation.copy()
            right_rot = self.right_rotation.copy()
        
        return left_pos, left_rot, right_pos, right_rot

    # Getters (not thread safe)
    def get_left_position(self):
        """Get position of first controller"""
        return self.left_position.copy()
    
    def get_right_position(self):
        """Get position of second controller"""
        return self.right_position.copy()
    
    def get_left_rotation(self):
        """Get rotation quaternion of first controller"""
        return self.left_rotation.copy()
    
    def get_right_rotation(self):
        """Get rotation quaternion of second controller"""
        return self.right_rotation.copy()
    
    # Setters (absolute)
    def set_left_position(self, position):
        """Set absolute position of first controller"""
        self.left_position = np.array(position)
    
    def set_right_position(self, position):
        """Set absolute position of second controller"""
        self.right_position = np.array(position)
    
    def set_left_rotation(self, rotation):
        """Set absolute rotation quaternion of first controller"""
        self.left_rotation = np.array(rotation)
        self.left_rotation = self.left_rotation / np.linalg.norm(self.left_rotation)  # Normalize quaternion
    
    def set_right_rotation(self, rotation):
        """Set absolute rotation quaternion of second controller"""
        self.right_rotation = np.array(rotation)
        self.right_rotation = self.right_rotation / np.linalg.norm(self.right_rotation)  # Normalize quaternion
    
    # Update methods (relative change)
    def update_left_position(self, delta_position):
        """Update position of first controller by adding delta"""
        self.left_position = self.left_position + np.array(delta_position)
    
    def update_right_position(self, delta_position):
        """Update position of second controller by adding delta"""
        self.right_position = self.right_position + np.array(delta_position)
    
    def update_left_rotation(self, delta_rotation):
        """Update rotation of first controller by multiplying current quaternion with delta"""
        self.left_rotation = self.multiply_quaternions(self.left_rotation, np.array(delta_rotation))
        self.left_rotation = self.left_rotation / np.linalg.norm(self.left_rotation)  # Normalize quaternion
    
    def update_right_rotation(self, delta_rotation):
        """Update rotation of second controller by multiplying current quaternion with delta"""
        self.right_rotation = self.multiply_quaternions(self.right_rotation, np.array(delta_rotation))
        self.right_rotation = self.right_rotation / np.linalg.norm(self.right_rotation)  # Normalize quaternion

    @staticmethod
    def multiply_quaternions(q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])


class JoyconInterface(BimanualTeleopInterface):
    """Interface for Nintendo Joy-Con controllers that inherits from BimanualTeleopInterface"""
    
    def __init__(self, joycon_script_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/joycon_reader.py"):
        super().__init__()
        
        # Constants for signal processing
        self.accel_scale = 0.0001  # Scale for acceleration
        self.accel_decay = 0.99    # Decay factor for velocity (simulates friction)
        self.dt = 0.01             # Time step (10ms)
        
        # Storage for Joy-Con data
        self.joycon_data = {
            "accel": None,
            "gyro": None,
            "accel1": None,
            "gyro1": None
        }
        self.data_lock = threading.Lock()
        
        # Start the Joy-Con data script as a subprocess
        self.joycon_script_path = joycon_script_path
        self.joycon_process = subprocess.Popen(
            ["python", self.joycon_script_path], 
            stdout=subprocess.PIPE, 
            stderr=sys.stderr, 
            text=True
        )
        
        # Launch thread to read from process
        self.reader_thread = threading.Thread(target=self._read_remote_joycon_data)
        self.reader_thread.daemon = True  # Thread will close when main program exits
        self.reader_thread.start()
    

    def _read_remote_joycon_data(self):
        """Read data from the Joy-Con subprocess"""
        while True:
            try:
                line = self.joycon_process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                    with self.data_lock:
                        self.update(data)
                        # print(self.joycon_data)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON: ", line)
                    print(f"Failed to parse JSON: {e}")
            except Exception as e:
                print(f"Error reading from subprocess: {e}")
                break

    def gyro_to_quaternion(self, gyro_data):
        """Convert gyro data to quaternion rotation delta"""
        # Scale gyro data
        scale = 0.01
        
        # Extract gyro values
        gyro_x = gyro_data[0] * scale
        gyro_y = gyro_data[1] * scale
        gyro_z = gyro_data[2] * scale
        
        # Calculate rotation angle (magnitude)
        angle = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2) * self.dt
        
        # Handle zero rotation case
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        # Normalize rotation axis
        axis_x = gyro_x / angle
        axis_y = gyro_y / angle
        axis_z = gyro_z / angle
        
        # Create quaternion [w, x, y, z]
        half_angle = angle / 2.0
        sin_half_angle = np.sin(half_angle)
        
        qw = np.cos(half_angle)
        qx = axis_x * sin_half_angle
        qy = axis_y * sin_half_angle
        qz = axis_z * sin_half_angle
        
        return np.array([qw, qx, qy, qz])

    def update(self, joycon_data):
        """Update position and rotation based on Joy-Con data"""        
        if not joycon_data or not any(joycon_data.values()):
            return
        
        # Process first Joy-Con data
        accel = joycon_data.get("accel")
        gyro = joycon_data.get("gyro")
        
        # Process second Joy-Con data
        accel1 = joycon_data.get("accel1")
        gyro1 = joycon_data.get("gyro1")
        
        # Convert to numpy arrays if data exists
        accel = np.array(accel) if accel else np.array([0.0, 0.0, 0.0])
        gyro = np.array(gyro) if gyro else np.array([0.0, 0.0, 0.0])
        accel1 = np.array(accel1) if accel1 else np.array([0.0, 0.0, 0.0])
        gyro1 = np.array(gyro1) if gyro1 else np.array([0.0, 0.0, 0.0])
        
        # Update rotation for first Joy-Con using gyro data
        if gyro is not None and np.any(gyro):
            delta_quat1 = self.gyro_to_quaternion(gyro)
            self.update_left_rotation(delta_quat1)
        
        # Update rotation for second Joy-Con using gyro data
        if gyro1 is not None and np.any(gyro1):
            delta_quat2 = self.gyro_to_quaternion(gyro1)
            self.update_right_rotation(delta_quat2)
        
        # Update position using accelerometer data for first Joy-Con
        if accel is not None and np.any(accel):
            # Apply rotation to accelerometer data to get world-space acceleration
            # Note: This is a simplified version - full IMU processing would be more complex
            world_accel1 = accel * self.accel_scale
            
            # Update velocity using acceleration
            self.velocity1 = self.velocity1 * self.accel_decay + world_accel1 * self.dt
            
            # Update position using velocity
            self.update_left_position(self.velocity1 * self.dt)
        
        # Update position using accelerometer data for second Joy-Con
        if accel1 is not None and np.any(accel1):
            world_accel2 = accel1 * self.accel_scale
            self.velocity2 = self.velocity2 * self.accel_decay + world_accel2 * self.dt
            self.update_right_position(self.velocity2 * self.dt)

    def close(self):
        """Properly close the Joy-Con subprocess"""
        if hasattr(self, 'joycon_process'):
            self.joycon_process.terminate()
            self.joycon_process.wait()


# Example usage
if __name__ == "__main__":
    try:
        # Create JoyconInterface instance
        joycon_interface = JoyconInterface()
        
        # Main loop
        while True:
            joycon_interface.update()
            
            # Example of accessing data
            if np.random.random() < 0.01:  # Only print occasionally to reduce spam
                print("Position 1:", joycon_interface.get_left_position())
                print("Position 2:", joycon_interface.get_right_position())
                print("Rotation 1:", joycon_interface.get_left_rotation())
                print("Rotation 2:", joycon_interface.get_right_rotation())
            
            time.sleep(0.01)  # Small delay to prevent CPU hogging
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if 'joycon_interface' in locals():
            joycon_interface.close()