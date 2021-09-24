import os
import shutil

emotions_dict = {0: "neutral", 1: "anger", 2: "contempt", 3: "disgust", 4: "fear", 5: "happy",
                             6: "sad", 7: "surprise"}

def get_emotion_from_txt_file (txt_file):
    with open(txt_file, "r") as file:
        string = file.readline()
        string = string.strip()
        emotion_number = int(string[0])
        emotion = emotions_dict[emotion_number]
    return emotion

def remove_DS_store_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith(".DS"):
                print(f"Removing File: {file}")
                try:
                    os.remove(os.path.join(root, file))
                except FileNotFoundError:
                    pass

def main():
    images_root_dir = r"C:\Users\Carmel\PycharmProjects\AIProject\CK+\cohn-kanade-images"
    labels_root_dir = r"C:\Users\Carmel\PycharmProjects\AIProject\CK+\Emotion"
    project_path = r"C:\Users\Carmel\PycharmProjects\AIProject"
    destination_dir = "CK_preprocessed"
    dest_path = os.path.join(project_path,destination_dir)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    for emotion in emotions_dict.values():
        dir = os.path.join(dest_path, emotion)
        if not os.path.exists(dir):
            os.mkdir(dir)
    human_class_dir = os.path.join(dest_path, "need_to_be_classified")
    os.mkdir(human_class_dir)


    remove_DS_store_files(images_root_dir)
    remove_DS_store_files(labels_root_dir)



    for sample,sample_label in zip(os.listdir(images_root_dir), os.listdir(labels_root_dir)): # list of S0005, S010
        sample_dir = os.path.join(images_root_dir,sample)
        sample_label_dir = os.path.join(labels_root_dir,sample_label)
        if len(os.listdir(sample_dir)) == 0:
            continue
        else:
            for sample_image_dir, sample_emotion_dir in zip(os.listdir(sample_dir), os.listdir(sample_label_dir)): # list of 001, 002,
                inner_sample_dir = os.path.join(sample_dir,sample_image_dir)
                inner_sample_label_dir = os.path.join(sample_label_dir,sample_emotion_dir)
                if len(os.listdir(inner_sample_dir)) == 0:
                    continue
                else:
                    images = os.listdir(inner_sample_dir)
                    if inner_sample_dir.endswith("001"):
                        neutral_face = images[0]
                        neutral_face_path = os.path.join(inner_sample_dir,images[0])
                        neutral_dir = os.path.join(dest_path,"neutral")
                        shutil.copyfile(neutral_face_path,os.path.join(neutral_dir,neutral_face))
                    peak_image = images[-1]
                    peak_image_path = os.path.join(inner_sample_dir,images[-1])
                    if len(os.listdir(inner_sample_label_dir)) == 0: # no label for the image
                        shutil.copyfile(peak_image_path,os.path.join(human_class_dir,peak_image))
                    else:
                        txt_file = os.path.join(inner_sample_label_dir ,os.listdir(inner_sample_label_dir)[0])
                        emotion = get_emotion_from_txt_file(txt_file)
                        emotions_destination_dir = os.path.join(dest_path,emotion)
                        shutil.copyfile(peak_image_path,os.path.join(emotions_destination_dir,peak_image))


if __name__ == '__main__':
    main()