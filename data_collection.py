""" 
Python script for custom data collection.
"""

# Importing libraries
import os                  # for handling file directories
import cv2                 # for video input and output as processed images
import numpy as np         # for handling large numerical values
import pandas as pd        # for processing csv file

# For saving images to file directories
DATA_DIR = './'                         # current directory
if not os.path.exists(DATA_DIR):        # checks for existence of data file in current directory and creates new file if not found
    os.makedirs(DATA_DIR)

# File size of each alphabet
number_of_classes = 26          # for each letter
dataset_size = 150              # number of images collected for each letter [customizable]

# Collecting data
cap = cv2.VideoCapture(0)                                               # starts infinite loop for camera feed of user's system 
for j in range(number_of_classes):                                      # loop for each letter in the alphabet
    if not os.path.exists(os.path.join(DATA_DIR, str(chr(j+97)))):      # creates new file directories for each letter if it doesn't exist in the current data file
        os.makedirs(os.path.join(DATA_DIR, str(chr(j+97))))

    print('Collecting data for letter: {}'.format(str(chr(j+97))))

    while True:
        ret, frame = cap.read()             
        cv2.putText(frame, "Press 'X' when ready!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 0), 2,
                    cv2.LINE_AA)                                       # text output on frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('x'):                                # waits for keyboard input to start
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(chr(j+97)) + "_" + str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()

image_data=[]
labels =[]

for j in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, str(chr(j+97)))

    for filename in os.listdir(class_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(class_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            image_flat = image.flatten()
            image_data.append(image_flat)
            labels.append(j)

image_data = np.array(image_data)
labels = np.array(labels)

mu, sigma = 0, 0.1 
noise = np.random.normal(mu, sigma, image_data.shape) 

df = pd.DataFrame(image_data)

df+=noise

df['label'] = labels

df.to_csv('sign_language.csv', index=False)

print("Data has been saved to 'sign_language.csv'")