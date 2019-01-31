
import os

root_path = "C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\openface_keras\\data"
root_dir_list = os.listdir(root_path)

print(root_dir_list)

for dir_name in root_dir_list:
    dir_list = os.listdir(root_path + os.sep + dir_name)
    c = 1
    for file_name in dir_list:
        current_file_path = root_path + os.sep + dir_name + os.sep + file_name

        new_file_name = file_name.replace(file_name, dir_name + "_" + str(c).zfill(4) + ".jpg")
        new_file_path = root_path + os.sep + dir_name + os.sep + new_file_name
        os.rename(current_file_path, new_file_path)
        c = c + 1


