# Main entry point to run the simulation loop

import carla
import cv2
import numpy as np
import random
import time

def main():
    client = None
    try:
        # 1. Connect to the CARLA server running locally
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        world = client.get_world()
        print("Successfully connected to CARLA server!")

        # 2. Get the blueprint library and spawn an ego-vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        
        # Get a random spawn point on the map
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego-vehicle: {vehicle.type_id}")

        # Set vehicle to autopilot for testing initial telemetry
        vehicle.set_autopilot(True)

        # 3. Attach a camera sensor to simulate your dashboard/vision input
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        
        # Position camera at the rear update (the earlier was just for example)
        camera_transform = carla.Transform(carla.Location(x=2.0,z=1.2), carla.Rotation(yaw=180))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 4. Define the live data callback function (Where your YOLOv8 / CV model will go)
        def process_image(image):
            # Convert CARLA raw image bytes to an OpenCV numpy array
            i = np.array(image.raw_data)
            i2 = i.reshape((720, 1280, 4)) # BGRA format
            frame = i2[:, :, :3]            # Extract BGR
            
            # --- COMPUTER VISION & EID WARNING HOOK ---
            # TODO: Insert your YOLOv8 model inference here:
            # results = model(frame)
            # Apply dual-variable warning gradient logic based on distance/hazards
            
            # Display the live 16:9 feed in a window representing your center console
            cv2.imshow("SpatialPark - 16:9 Center Console Display", frame)
            cv2.waitKey(1)

        # Listen to the camera stream
        camera.listen(lambda image: process_image(image))

        # Keep the script running to maintain simulation
        print("Simulation running. Press Ctrl+C in terminal to exit.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping simulation and cleaning up actors...")
    finally:
        # Clean up actors so they don't linger in CARLA memory
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()