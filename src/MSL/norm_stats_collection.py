import time
import json
import numpy as np
from xarm.wrapper import XArmAPI

# Connect to your XArm (replace with your robot's IP)
arm = XArmAPI('192.168.1.219')  
arm.connect()

print("Logging XArm joint + gripper positions...")
samples = []

try:
    while True:
        # Read joint positions (6 joints)
        joints = arm.get_servo_angle(is_radian=True)[1][:6]  # returns tuple (ret_code, data)
        # Read gripper position (if you have a Robotiq gripper, might be different API)
        gripper = arm.get_gripper_position()[1] if hasattr(arm, "get_gripper_position") else 0.0

        state = np.array(joints + [gripper])
        samples.append(state)

        print(f"Sampled: {state}")
        time.sleep(0.05)  # sample at ~20 Hz

except KeyboardInterrupt:
    print("Stopping logging and computing statistics...")

# Compute mean and std
data = np.array(samples)
mean = data.mean(axis=0)
std = data.std(axis=0)

norm_stats = {"mean": mean.tolist(), "std": std.tolist()}

# Save to JSON
with open("xarm_norm_stats.json", "w") as f:
    json.dump(norm_stats, f, indent=4)

print("Saved norm stats to xarm_norm_stats.json")
print("Mean:", mean)
print("Std:", std)