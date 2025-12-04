import time
import json
import numpy as np
from xarm.wrapper import XArmAPI

# Connect to your XArm (replace with your robot's IP)
arm = XArmAPI('192.168.1.219')  
arm.connect()

time.sleep(0.5)
print(arm.get_gripper_position()[1])
time.sleep(5)
print("Logging XArm joint + gripper positions...")
states = []
actions = []
prev_joints = arm.get_servo_angle(is_radian=True)[1][:6]
prev_gripper = (arm.get_gripper_position()[1] - 850) / -860

try:
    while True:
        # Read joint positions (6 joints)
        joints = arm.get_servo_angle(is_radian=True)[1][:6]  # returns tuple (ret_code, data)
        # Read gripper position (if you have a Robotiq gripper, might be different API)
        gripper = (arm.get_gripper_position()[1] - 850) / -860

        state = np.array(joints + [gripper])
        action = state - np.array(prev_joints + [prev_gripper])
        states.append(state)
        actions.append(action)
        prev_joints = joints
        prev_gripper = gripper
        print(f"Sampled: {state}")
        time.sleep(0.02)  # sample at ~20 Hz

except KeyboardInterrupt:
    print("Stopping logging and computing statistics...")

# Compute mean and std
data1 = np.array(actions)
data2 = np.array(states)

norm_stats = {
    "norm_stats": {
        "actions": 
        {
            "mean": data1.mean(axis=0).tolist(),
            "std": data1.std(axis=0).tolist(),
            "q01": np.quantile(data1, 0.01, axis=0).tolist(),
            "q99": np.quantile(data1, 0.99, axis=0).tolist(),
        },
        "state": 
        {
            "mean": data2.mean(axis=0).tolist(),
            "std": data2.std(axis=0).tolist(),
            "q01": np.quantile(data2, 0.01, axis=0).tolist(),
            "q99": np.quantile(data2, 0.99, axis=0).tolist(),
        }
    }
}

# Save to JSON
with open("norm_stats.json", "w") as f:
    json.dump(norm_stats, f, indent=4)

print("Saved norm stats to norm_stats.json")
print("state mean:", data2.mean(axis=0))
print("action mean:", data1.mean(axis=0))

print("state std:", data2.std(axis=0))
print("action std:", data1.std(axis=0))