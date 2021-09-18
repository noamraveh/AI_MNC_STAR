import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PreProcess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = pd.read_csv(dataset_name + ".csv")
        self.one_hot_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise",
                             6: "Neutral"}
        self.df = self.df.replace({"emotion": self.one_hot_dict})

    def show_split(self):
        check = self.df.groupby('emotion').count()
        pie_chart_dict = dict(zip(list(check.index), list(check.pixels)))
        print(pie_chart_dict)
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


def generate_np_arrays(df):
    original_arrays = df.image
    images = []
    for sample in original_arrays:
        image = np.array(sample.split(), dtype="float32")
        image = image.reshape(48, 48)
        images.append(image)
    df["image"] = images
    return df


def main_2():
    fer13 = PreProcess("FER2013")
    fer13.show_split()
    fer13.show_pic_of_each_emotion()


if __name__ == '__main__':
    main_2()
