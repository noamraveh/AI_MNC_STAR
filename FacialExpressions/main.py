import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

def generate_np_arrays(df):
    original_arrays = df.image
    images = []
    for sample in original_arrays:
        image = np.array(sample.split(), dtype="uint8")
        image = image.reshape(48, 48)
        images.append(image)
    df["image"] = images
    return df


def main():
    faces = pd.read_csv("faces.csv")[['label', 'image']]
    faces = generate_np_arrays(faces)
    train, test = train_test_split(faces, test_size=0.3, random_state=10)
    X_train = train['image']
    X_test = test['image']
    Y_train = train['label']
    Y_test = test['label']
    im = plt.imshow(X_train[0], cmap='gray')
    plt.show()


    print("HELLO")


if __name__ == '__main__':
    main()

