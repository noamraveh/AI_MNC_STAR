import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key


def create_ck_csv(ck_root_dir_path):
    csv_filename = "CK+"
    emotions_dict = {0: "neutral", 1: "anger", 2: "contempt", 3: "disgust", 4: "fear", 5: "happy",
                     6: "sad", 7: "surprise"}
    emotion = []
    pixels = []
    for emotion_dir in os.listdir(ck_root_dir_path):
        emotion_number = get_key(emotions_dict, emotion_dir)
        emotion_dir_path = os.path.join(ck_root_dir_path, emotion_dir)
        for image in os.listdir(emotion_dir_path):
            im = Image.open(os.path.join(emotion_dir_path, image))
            im_pixels_list = [str(pixel) for pixel in im.getdata()]
            im_pixels = " ".join(im_pixels_list)
            emotion.append(emotion_number)
            pixels.append(im_pixels)

    df = pd.DataFrame(list(zip(emotion, pixels)),
                      columns=['emotion', 'pixels'])
    df.to_csv(csv_filename + ".csv")


class PreProcess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = pd.read_csv(dataset_name + ".csv")
        if dataset_name == "FER2013":
            self.one_hot_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise",
                                 6: "Neutral"}
        if dataset_name == "CK+":
            self.one_hot_dict = {0: "neutral", 1: "anger", 2: "contempt", 3: "disgust", 4: "fear", 5: "happy",
                                 6: "sad", 7: "surprise"}
        self.df = self.df.replace({"emotion": self.one_hot_dict})

    def show_split(self):
        check = self.df.groupby('emotion').count()
        pie_chart_dict = dict(zip(list(check.index), list(check.pixels)))
        fig1, ax1 = plt.subplots()
        total = sum(pie_chart_dict.values())
        ax1.pie(pie_chart_dict.values(), labels=pie_chart_dict.keys(), autopct=lambda p: '{:.0f}'.format(p * total / 100))
        plt.title(f"Distribution of Emotions in {self.dataset_name}")
        ax1.axis('equal')
        plt.show()

    def show_pic_of_each_emotion(self):
        rows = 2
        cols = 4
        axes = []
        fig = plt.figure()
        plt.title(f"{self.dataset_name} Examples")
        plt.axis("off")
        img_examples = []
        for emotion_type in self.one_hot_dict.values():
            img_examples.append(self.df[self.df.emotion == emotion_type].iloc[3]["pixels"])

        for i, emotion in enumerate(self.one_hot_dict.values()):
            b = img_examples[i]
            image = np.array(b.split(), dtype="float32")
            image = image.reshape(48, 48)
            axes.append(fig.add_subplot(rows, cols, i+1))
            subplot_title = emotion
            axes[-1].set_title(subplot_title)
            plt.imshow(image, cmap="gray")
        fig.tight_layout()
        plt.show()


def main_2():
    fer13 = PreProcess("FER2013")
    fer13.show_split()
    fer13.show_pic_of_each_emotion()

    # ck_root_dir_path = r"C:\Users\Carmel\PycharmProjects\AIProject\CK_preprocessed"
    # create_ck_csv(ck_root_dir_path)
    ck = PreProcess("CK+")
    ck.show_split()


if __name__ == '__main__':
    main_2()
