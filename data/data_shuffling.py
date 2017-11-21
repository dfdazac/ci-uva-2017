import numpy as np

files = ["aalborg.csv", "alpine-1.csv", "f-speedway.csv"]

# Load first file
data = np.loadtxt(files[0], skiprows=1, delimiter=",")

# Load remaining files
for i in range(1, len(files)):
    data = np.vstack((data, np.loadtxt(files[i], skiprows=1, delimiter=",")))

np.random.shuffle(data)

train_perc = 0.7
valid_perc = 0.2
test_perc = 1 - (train_perc + valid_perc)

train_i = int(len(data) * train_perc)
valid_i = train_i + int(len(data) * valid_perc)

train_data = data[:train_i]
valid_data = data[train_i:valid_i]
test_data = data[valid_i:]

np.savetxt("train.csv", train_data, delimiter=",")
np.savetxt("valid.csv", valid_data, delimiter=",")
np.savetxt("test.csv", test_data, delimiter=",")