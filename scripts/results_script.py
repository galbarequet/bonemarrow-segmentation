import os
import numpy as np
import seaborn as sn
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
# Relative importing
import sys
sys.path.append(os.getcwd())
from inference import plot_param_dist
from skimage.io import imsave


prediction_path = None


def print_menu():
    print('Menu:')
    print('1 - show confusion matrix')
    print('2 - insert prediction folder path')
    print('3 - calculate learning graph')
    print('4 - calculate dsc stats')
    print('5 - create dsc graphs')
    print('6 - show avg. confusion matrix')
    print('q - quit')


def normalize_confusion_matrix(confusion_mat):
    confusion_mat = confusion_mat[:][:]
    for i in range(len(confusion_mat)):
        norm_factor = sum(confusion_mat[i])
        for j in range(len(confusion_mat[i])):
            confusion_mat[i][j] /= norm_factor
    return confusion_mat


def print_confusion_matrix_graph(confusion_mat, normalize=False):
    if normalize:
        confusion_mat = normalize_confusion_matrix(confusion_mat)
    labels = ['Bone', 'Fat', 'Other Tissue', 'Background']
    df_cm = pd.DataFrame(confusion_mat, index=labels, columns=labels)
    plt.figure(figsize=(7, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


def get_confusion_matrix(path_con_mat=None, print_graph=False, normalize_mat=True):
    if path_con_mat is None:
        path_con_mat = input('Insert path to confusion matrix:\n')
    with open(path_con_mat, 'r', encoding='utf-8') as f:
        confusion_mat = json.load(f)
    if print_graph:
        print_confusion_matrix_graph(confusion_mat, normalize=normalize_mat)
    return confusion_mat


def get_avg_normalized_confusion_matrix(print_graph=False):
    global prediction_path
    if prediction_path is None:
        prediction_path = input('Insert path to predictions dir:\n')
    name_to_confusion_matrix = {}
    for i, filename in enumerate(os.listdir(prediction_path)):
        with open(os.path.join(prediction_path, filename, f'stats - {filename}.json'), 'r', encoding='utf-8') as f:
            name_to_confusion_matrix[filename] = normalize_confusion_matrix(json.load(f))
    avg_confusion_matrix = np.average(list(name_to_confusion_matrix.values()), axis=0)
    if print_graph:
        # Normalizing just in case
        print_confusion_matrix_graph(avg_confusion_matrix, normalize=True)
    return avg_confusion_matrix


def change_pred_dir():
    global prediction_path
    prediction_path = input('Insert path to prediction folder:\n')


def graph(y_values, y_label='', x_label='', show_graph=True, legend=None):
    x = range(1, len(y_values) + 1)
    plt.plot(x, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(legend)
    if show_graph:
        plt.show()


def learning_graph():
    path_log_output = input('Insert path to output file from the training process:\n')
    num_epoch = int(input('Insert number of epochs to display (-1 for all epochs):\n'))

    fat_dsc = []
    bone_dsc = []
    other_tissue_dsc = []
    background_dsc = []
    density_error = []
    validation_loss = []
    with open(path_log_output, 'r', encoding='utf-8') as f:

        for line in f:
            if 'fat dsc' in line:
                fat_dsc.append(float(line.split()[-1]))
            if 'bone dsc' in line:
                bone_dsc.append(float(line.split()[-1]))
            if 'tissue dsc' in line:
                other_tissue_dsc.append(float(line.split()[-1]))
            if 'background dsc' in line:
                background_dsc.append(float(line.split()[-1]))
            if 'bone density error' in line:
                density_error.append(float(line.split()[-1][:-1]))
            if 'validation loss' in line:
                validation_loss.append(float(line.split()[-1]))
            if (num_epoch != -1 and len(fat_dsc) >= num_epoch and
               len(validation_loss) >= num_epoch and len(bone_dsc) >= num_epoch):
                break

    graph(bone_dsc, show_graph=False)
    graph(fat_dsc, show_graph=False)
    graph(other_tissue_dsc, show_graph=False)
    graph(background_dsc, y_label='Validation DSC', x_label='epoch',
          legend=['Bone DSC', 'Fat DSC', 'Other Tissue DSC', 'Background DSC'])
    graph(density_error, y_label='Bone Density Relative Error Percentage', x_label='epoch')
    graph(validation_loss, y_label='Validation Loss', x_label='epoch')


def calc_dsc_from_cm(cm, index):
    np_cm = np.array(cm)
    return (2 * np_cm[index, index]) / (sum(np_cm[index, :]) + sum(np_cm[:, index]))


def get_dsc_from_json_file(file_name):
    """
        The function assumes prediction_path is set and doesn't check it.
        
        Return: bone_dsc, fat_dsc
    """
    cm_path = os.path.join(prediction_path, file_name, 'stats - {}.json'.format(file_name))
    try:
        cm = get_confusion_matrix(path_con_mat=cm_path)
    except FileNotFoundError:
        print("Haven't found json file in {}".format(file_name))
        return None, None
    bone_dsc = calc_dsc_from_cm(cm, 0)
    fat_dsc = calc_dsc_from_cm(cm, 1)
    return bone_dsc, fat_dsc


def calculate_dsc_for_all():
    """
        Format of output file: image name,bone dsc,fat dsc
    """
    global prediction_path
    if prediction_path is None:
        print('please set the path to prediction folder first.')
        return
    path_stats_file = input('Insert path to where to save the stats to.\n')
    with open(path_stats_file, 'w+') as write_f:
        for folder in os.listdir(prediction_path):
            bone_dsc, fat_dsc = get_dsc_from_json_file(folder)
            if bone_dsc is None:
                continue
            write_f.write("{},{},{}\n".format(folder, bone_dsc, fat_dsc))


def create_dsc_graphs():
    if prediction_path is None:
        print('please set the path to prediction folder first.')
        return
    bone_dsc_list = []
    fat_dsc_list = []
    file_names = []
    while True:
        print('Menu:')
        print('a - insert new file to add to graph')
        print('p - save graphs')
        print('q - exit')
        input_op = input()
        if input_op == 'a':
            file_name = input('insert the image name.\n')
            bone_dsc, fat_dsc = get_dsc_from_json_file(file_name)
            if bone_dsc is None:
                continue
            bone_dsc_list.append(bone_dsc)
            fat_dsc_list.append(fat_dsc)
            file_names.append(file_name)
            print('added {} to graph'.format(file_name))
        elif input_op == 'p':
            save_folder_path = input('Insert path to folder to save the graphs to.\n')
            os.makedirs(save_folder_path, exist_ok=True)
            dsc_bone_dist_plot = plot_param_dist(dict(zip(file_names, bone_dsc_list)))
            imsave(os.path.join(save_folder_path, 'dsc_bone.png'), dsc_bone_dist_plot)
            dsc_fat_dist_plot = plot_param_dist(dict(zip(file_names, fat_dsc_list)))
            imsave(os.path.join(save_folder_path, 'dsc_fat.png'), dsc_fat_dist_plot)
        elif input_op == 'q':
            return
        else:
            print('Invalid command')


if __name__ == '__main__':
    while True:
        print_menu()
        input_op = input()
        if input_op == '1':
            get_confusion_matrix(print_graph=True, normalize_mat=True)
        elif input_op == '2':
            change_pred_dir()
        elif input_op == '3':
            learning_graph()
        elif input_op == '4':
            calculate_dsc_for_all()
        elif input_op == '5':
            create_dsc_graphs()
        elif input_op == '6':
            get_avg_normalized_confusion_matrix(print_graph=True)
        elif input_op == 'q':
            exit()
        else:
            print('Invalid command')
