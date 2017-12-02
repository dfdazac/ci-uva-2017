import numpy as np
import matplotlib.pyplot as plt

# Acceleration, brake, steering
column_to_values = {}
# Acceleration
column_to_values[0] = ([0, 1], 0.4)
# Brake
column_to_values[1] = ([0, 1], 0.4)
# Steering
column_to_values[2] = ([-1, -3/4, -1/3, 0, 1/3, 3/4, 1], 0.1)

in_file_name = "all_sensors_all_controls.csv"
out_file_name = "blackboard_quantized.csv"
input_file = open(in_file_name)
output_file = open(out_file_name, "w")

# Write the header (and add the missing column label)
output_file.write(input_file.readline()[:-1] + ",TRACK_EDGE_18\n")

# Process the rest of the lines
for line in input_file:
    line_data = line.strip().split(",")
    
    # Process columns that need to be quantized
    for col in column_to_values:
        value = float(line_data[col])

        # Find the corresponding discrete value
        discrete_values = np.array(column_to_values[col][0])
        closest_value_idx = np.argmin(np.abs(discrete_values - value))
        line_data[col] = str(closest_value_idx)

    # Regroup the data and write to new file
    output_file.write(",".join(line_data) + "\n")

input_file.close()
output_file.close()

# Check
original = np.loadtxt(in_file_name, delimiter=",", skiprows=1)
processed = np.loadtxt(out_file_name, delimiter=",", skiprows=1)
titles = ["Acceleration", "Brake", "Steering"]
for i in range(3):    
    plt.subplot(3, 1, i+1)
    plt.plot(original[:,i], lw=0.5)
    plt.plot(processed[:, i], "r.", markersize=1)
    plt.title(titles[i])
plt.show()

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(processed[:, i])
    plt.title(titles[i])
plt.show()
