from coppeliasim_interface import CoppeliaSimInterface 
from coppeliaUI import CoppeliaSimUI 
import numpy as np
from enum import Enum
import time
import os
from yolov5_detector import YoloDetector
from path_planning import CubicPathPlanner
from speech_recognizer import SpeechRecognizer
import cv2

q_release = np.deg2rad([0, 30, 0, -90, 0, 120, 0])    # Position to release the object
q_vision = np.deg2rad([90, 0, 0, -90, 0, 90, 0])    # Position where the image is captured
q_object = np.zeros(7)


class RobotState(Enum):
    REACH_VISION = 1        # Reach the configuration where the piucture must be taken
    LISTEN_MIC = 2          # Listen the microphone channel 
    IMAGE_CAPTURE = 3       # Capture vision sensor image, analayse it and compute IK
    REACH_OBJECT = 4        # Reach the object that must be grasped
    GRASP = 5               # Grasp object
    REACH_RELEASE = 6       # Reach the configuration where the object must be released
    RELEASE = 7             # Release the object
    DELETE = 8              # Delete object after release



if __name__ == "__main__":

    sp_rec_object_names = ["COLTELLO", "FORCHETTA", "TAZZA", "CUCCHIAIO", "MEDICINA", "BICCHIERE"]
    sp_rec_translation_dict = {'COLTELLO': "KNIFE", 'FORCHETTA': "FORK", 'TAZZA':'MUG', 'BICCHIERE':'CUP', 'CUCCHIAIO': 'SPOON', 'MEDICINA': 'MEDICINE'}
    sp_rec = SpeechRecognizer(object_names=sp_rec_object_names)

    joint_names = [f"/joint{i+1}" for i in range(0,7)]
    vision_sensor_names = ["/rgbd_sensor"]
    coppelia_int = CoppeliaSimInterface(real_time=False)
    
    
    coppelia_int_ui = CoppeliaSimUI()                         # coppelia user interface
    coppelia_ui, id_sim_window = coppelia_int_ui.init_window()


    joint_handles = coppelia_int.get_handles(joint_names)
    gripper_handle = coppelia_int.get_handles(["/openCloseJoint"])[0]
    rgbd_sensor_handle = coppelia_int.get_handles(vision_sensor_names)[0]
    target_sphere_handle = coppelia_int.get_handles(["/target_sphere"])[0] 

    base_handle = coppelia_int.get_handles(["/Franka"])[0]                     # For IK - the base of the robot
    tip_handle = coppelia_int.get_handles(["/tip"])[0]                         # For IK
    target_handle = coppelia_int.get_handles(["/target_dummy"])[0]             # For IK

    res_x, res_y, angle_x, angle_y = coppelia_int.get_vision_sensor_params(rgbd_sensor_handle)
    coppelia_T_yolo = np.array([[-1, 0, res_x - 1], 
                                [0, 1, 0],
                                [0, 0, 1]])     # This matrix is static!

    weights_path = os.path.join("weights", "best_v2.pt")
    yolo_path = os.path.join("src", "yolov5")
    class_names = ["Cup", "fork", "Knife", "Medicine", "Mug", "Spoon"]
    labelled_images_dir = os.path.join("labelled_images")
    abs_path_labelled_imges =os.path.abspath("labelled_images") 
    
    detector = YoloDetector(yolo_path, weights_path, class_names)
    
    coppelia_int.init_inverse_kinematics(joint_handles=joint_handles, tip_handle=tip_handle, target_handle=target_handle, base_handle=base_handle)

    error_threshold = 0.01 
    motion_time = 6     # Motion duration for the path planner
    simulation_duration = 200
    
    grasp_count = 0
    grasp_iterations = 100
    gripper_open_vel = 0.02
    gripper_close_vel = -0.02 # [m/s]
    
    release_count = 0
    release_iterations = 100
    
    
    capture_count = 0 
    object_to_grasp_name = ""

    robot_state = RobotState.REACH_VISION
    
    q = coppelia_int.get_joint_positions(joint_handles)
    planner = CubicPathPlanner(q_start=q, q_finish=q_vision.tolist(), t_start=0, t_finish=motion_time)
    

    coppelia_int.start_simulation()
    while(coppelia_int.get_simulation_time() <= simulation_duration):
        #print(f"Time:{coppelia_int.get_simulation_time()} - Current state: {robot_state}")
        
        if robot_state == RobotState.REACH_VISION:
            q = np.array(coppelia_int.get_joint_positions(joint_handles))
            q_target = planner.calc_configuration(time=coppelia_int.get_simulation_time())            
            coppelia_int.set_joint_target_positions(joint_handles=joint_handles, target_positions=q_target)
           
            if np.linalg.norm(q - q_vision) < error_threshold:
                robot_state = RobotState.LISTEN_MIC
        
        elif robot_state == RobotState.LISTEN_MIC:
            result, italian_name, stop_simulation = sp_rec.request_object(coppelia_int_ui, coppelia_ui, id_sim_window)
            if result:
                left = coppelia_int.wait_sim(3) # for video
                object_to_grasp_name = sp_rec_translation_dict[italian_name]
                left = coppelia_int.wait_sim(3)
                print(f"[INFO] Vision will search for {object_to_grasp_name}")
                coppelia_int_ui.set_label(coppelia_ui,f"[INFO] Vision will search for {object_to_grasp_name}",id_sim_window)
                robot_state = RobotState.IMAGE_CAPTURE
            elif not result and stop_simulation:
                print("[INFO] You asked to stop the simulation.")
                coppelia_int_ui.set_label(coppelia_ui,"[INFO] You asked to stop the simulation.",id_sim_window)
                break
            else:
                print("[ERROR] The object you asked for is not in my database. Please request another object.")
                coppelia_int_ui.set_label(coppelia_ui,"[ERROR] The object you asked for is not in my database. Please request another object.",id_sim_window)

        elif  robot_state == RobotState.IMAGE_CAPTURE:
            left = coppelia_int.wait_sim(3) # for video
            world_T_camera = coppelia_int.get_object_pose_matrix(rgbd_sensor_handle, coppelia_int.sim.handle_world)  

            # Capture the RGB image and the depth map from the vision sensor
            coppelia_int.set_object_position(target_sphere_handle, [0,0,0])     # Move the sphere away from the camera
            coppelia_int.enable_vision_sensor(rgbd_sensor_handle)           # This must be called otherwhise the camera is not updated (investigate this in the docs)
            rgb_image = cv2.flip(coppelia_int.get_rgb_image(rgbd_sensor_handle), 1)
            depth_image = cv2.flip(coppelia_int.get_depth_image(rgbd_sensor_handle, res_x, res_y), 1)

            pred_results = detector.inference(rgb_image, res_x, res_y)
            detector.save_results(rgb_image, pred_results, os.path.join(labelled_images_dir, f"detect_{capture_count}.png"))
            coppelia_int_ui.set_img(coppelia_int,coppelia_ui,id_sim_window,os.path.join(abs_path_labelled_imges, f"detect_{capture_count}.png"))
            capture_count += 1

            object_detected = False
            object_to_grasp_index = None
    
            for idx in range(len(pred_results)):
                pixel_yolo = list(pred_results[idx]["midpoint"])
                pixel_coppelia = coppelia_int.transform_pixel_coord(pixel_yolo, coppelia_T_yolo)
                p = coppelia_int.pixel_to_3d_world(pixel_coppelia, depth_image, world_T_camera, res_x, res_y, angle_x, angle_y)
                pred_results[idx] ["world_coord"] = p.copy()
    
                if pred_results[idx]["class_name"].upper() == object_to_grasp_name.upper():
                    object_detected = True
                    object_to_grasp_index = idx
                    break
            
            if object_detected:
                # The desired object is among the ones detected, perform IK to find the configuration for the robot
                coppelia_int.set_object_position(target_sphere_handle, pos=pred_results[object_to_grasp_index]["world_coord"])           # Set the target dummy position in the simulation
                q_object = np.array(coppelia_int.perform_inverse_kinematics(joint_handles=joint_handles, tip_handle=tip_handle, 
                                                                            target_handle=target_handle, base_handle=base_handle))
                # Initialize the path planner to reach the object
                q = np.array(coppelia_int.get_joint_positions(joint_handles))
                t_start = coppelia_int.get_simulation_time()
                t_finish = t_start + motion_time
                planner = CubicPathPlanner(q_start=q, q_finish=q_object, t_start=t_start, t_finish=t_finish)
            
                # Change the state so that the robot can reach the object
                robot_state = RobotState.REACH_OBJECT
            else:
                # The desired object is not among the ones detected, go back to listen state
                print(f"[WARNING] Desired object ({object_to_grasp_name.upper()}) not found! ")
                coppelia_int_ui.set_label(coppelia_ui,f"[WARNING] Desired object ({object_to_grasp_name.upper()}) not found! ",id_sim_window)

                robot_state = RobotState.LISTEN_MIC

        elif  robot_state == RobotState.REACH_OBJECT:
            # The robot must reach the configuration at which the object to pick is located 
            q = np.array(coppelia_int.get_joint_positions(joint_handles))
            q_target = planner.calc_configuration(time=coppelia_int.get_simulation_time())            
            coppelia_int.set_joint_target_positions(joint_handles=joint_handles, target_positions=q_target)
            # If the object has been reached, change he state to grasp
            if np.linalg.norm(q - q_object) < error_threshold:
                robot_state = RobotState.GRASP

        elif  robot_state == RobotState.GRASP:
            # Close the gripper to grasp the object
            coppelia_int.set_joint_target_velocities([gripper_handle], [gripper_close_vel])
            # Initialize the path planner to reach the releas position
            q = np.array(coppelia_int.get_joint_positions(joint_handles))
            t_start = coppelia_int.get_simulation_time()
            t_finish = t_start + motion_time
            q_via = [q_vision.tolist()]
            t_via = [t_start+motion_time/2]
            planner = CubicPathPlanner(q_start=q, q_finish=q_release.tolist(), t_start=t_start, t_finish=t_finish, q_via=q_via, t_via=t_via)
            # Change the state so that the releas position is reached
            if grasp_count == grasp_iterations:
                grasp_count = 0
                robot_state = RobotState.REACH_RELEASE
            else:
                grasp_count +=1
        elif  robot_state == RobotState.REACH_RELEASE:
            # The robot must reach the configuration at which the object to pick is located 
            q = np.array(coppelia_int.get_joint_positions(joint_handles))
            q_target = planner.calc_configuration(time=coppelia_int.get_simulation_time())            
            coppelia_int.set_joint_target_positions(joint_handles=joint_handles, target_positions=q_target)
           
            # If the object has been reached, change he state to grasp
            if np.linalg.norm(q - q_release) < error_threshold:
                robot_state = RobotState.RELEASE

        elif  robot_state == RobotState.RELEASE:
            # Open the gripper to release the object
            if release_count == release_iterations:
                release_count = 0
                coppelia_int.set_joint_target_velocities([gripper_handle], [gripper_open_vel])
                robot_state = RobotState.DELETE

                q = np.array(coppelia_int.get_joint_positions(joint_handles))
                t_start = coppelia_int.get_simulation_time()
                t_finish = t_start + motion_time
                planner = CubicPathPlanner(q_start=q, q_finish=q_vision.tolist(), t_start=t_start, t_finish=t_finish)
            else:
                release_count += 1
        elif robot_state == RobotState.DELETE: 
            # function for delete object in scene
            left = coppelia_int.wait_sim(0.50)
            coppelia_int.remove_obj(object_to_grasp_name)
            robot_state = RobotState.REACH_VISION
        
        coppelia_int.step()
        # time.sleep(coppelia_int.get_simulation_timestep())


    coppelia_int.stop_simulation()
