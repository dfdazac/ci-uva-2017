The following files were provided by the course staff. They consist of the controller actions (acceleration, brake and steering) as well as sensor data (speed, distance from center, angle to track and 19 rangefinders):
    - aalborg.csv
    - alpine-1.csv
    - f-speedway.csv

The data_shuffling.py script takes these three files and merge them into a matrix that is shuffled and divided into training, validation and testing data sets. They are saved as
    - train.csv
    - valid.csv
    - test.csv

Data was recorded comprising the distance from the center of the track and the steering produced by the basic PID steering controller AND the speed determined by the SOM. This data was saved in the center_steering folder and merged into a single file

Data of the car in three different positions with respect to the center of the track (left, center, right) was recorded for a single track. This is used to train a self-organizing map. The resulting data files are saved in the edges folder, and all of them were merged into a single file:
    - total.csv


