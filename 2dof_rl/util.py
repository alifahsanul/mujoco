import json
import numpy as np
import mujoco
import cv2

def load_json_variables(json_path='global_var.json'):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def add_text_frame(texts, frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, t in enumerate(texts):
        cv2.putText(frame_bgr, t, (10, 30 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def randomize_ball_position(mymodel, json_vars, seed=None, print_info=False):
    if seed is not None:
        np.random.seed(seed)
    
    radius = np.random.uniform(json_vars['robot_min_sweep_radius'], json_vars['robot_max_sweep_radius'])
    angle = np.random.uniform(np.pi + 0.5,  2 * np.pi - 0.5)
    ball_pos_x = radius * np.cos(angle)
    ball_pos_y = radius * np.sin(angle)

    if print_info:
        print(radius, angle, ball_pos_x, ball_pos_y)
        print('Original ball position:', mymodel.body_pos[mujoco.mj_name2id(mymodel, mujoco.mjtObj.mjOBJ_BODY, 'ball')])
        print('New ball position:', [ball_pos_x, ball_pos_y])
    mymodel.body_pos[mujoco.mj_name2id(mymodel, mujoco.mjtObj.mjOBJ_BODY, 'ball')][0] = ball_pos_x
    mymodel.body_pos[mujoco.mj_name2id(mymodel, mujoco.mjtObj.mjOBJ_BODY, 'ball')][1] = ball_pos_y
    return mymodel