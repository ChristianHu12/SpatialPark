# Main entry point to run the simulation loop

import carla
import cv2
import numpy as np
import random
import time
import traceback

from src.vision_processor import VisionProcess

# Global thread-safe container for the latest camera frame
latest_frame = None

def main():
    global latest_frame
    client = None
    camera = None
    vehicle = None
    
    try:
        # 1. Connect to the CARLA server running locally
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Successfully connected to CARLA server!")

        # 2. Get the blueprint library and spawn an ego-vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego-vehicle: {vehicle.type_id}")

        control = carla.VehicleControl()

        # 3. Attach a camera sensor with optimized 16:9 resolution (960x540)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '960')
        camera_bp.set_attribute('image_size_y', '540')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=-2.0, z=1.4), carla.Rotation(yaw=180))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Initialize the processor
        vision_proc = VisionProcess(width=960, height=540)

        # Initialize with placeholder text so the window never starts gray
        latest_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(latest_frame, "Waiting for CARLA stream...", (180, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # CARLA Callback: Process data safely and catch background thread exceptions
        def process_image(image):
            global latest_frame
            try:
                frame = vision_proc.process_carla_image(image)
                latest_frame = vision_proc.apply_cv_pipline(frame)
            except Exception as e:
                print("CRASH inside camera callback:")
                traceback.print_exc()

        camera.listen(lambda image: process_image(image))

        # Create window on the MAIN thread
        cv2.namedWindow("SpatialPark - 16:9 Center Console Display", cv2.WINDOW_NORMAL)

        # Keep track of persistent controls
        current_throttle = 0.0
        current_steer = 0.0
        current_brake = 0.0
        is_reverse = False

        print("Simulation running. Use W/S for throttle/brake, A/D for steering, R to toggle reverse. Press Q to exit.")
        while True:
            # Render the latest frame safely on the main thread
            if latest_frame is not None:
                cv2.imshow("SpatialPark - 16:9 Center Console Display", latest_frame)

            key = cv2.waitKey(1) & 0xFF  # Fast 1ms poll keeps UI responsive
            
            if key == ord('q') or key == ord('Q') or key == 27: # Press 'Q' or ESC to quit
                break

            # Toggle Reverse Gear with 'R'
            if key == ord('r') or key == ord('R'):
                is_reverse = not is_reverse
                print(f"Gear mode: {'REVERSE (R)' if is_reverse else 'DRIVE (D)'}")

            # Adjust controls based on keypresses
            if key == ord('w') or key == ord('W'):
                current_throttle = 0.4  # Gentle, controllable speed
                current_brake = 0.0
            elif key == ord('s') or key == ord('S'):
                current_throttle = 0.0
                current_brake = 0.8     # Firm brake
            elif key == ord('x') or key == ord('X'): # Coast / Neutral
                current_throttle = 0.0
                current_brake = 0.0

            if key == ord('a') or key == ord('A'):
                current_steer = -0.3    # Gentle steering angle
            elif key == ord('d') or key == ord('D'):
                current_steer = 0.3     # Gentle steering angle
            elif key == ord('c') or key == ord('C'): 
                current_steer = 0.0     # Re-center steering

            # Apply controls to vehicle
            control.throttle = current_throttle
            control.steer = current_steer
            control.brake = current_brake
            control.reverse = is_reverse
            
            vehicle.apply_control(control)
            time.sleep(0.02)  # Stable frame timing

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        print("Cleaning up actors quickly...")
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if vehicle is not None:
                vehicle.destroy()
        except Exception as e:
            print(f"Cleanup note: {e}")
        cv2.destroyAllWindows()
        print("Exit complete.")

if __name__ == '__main__':
    main()