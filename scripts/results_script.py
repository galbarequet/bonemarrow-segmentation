import os
import numpy as np
import seaborn as sn
import pandas as pd
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import pickle
import utils
# Relative importing
import sys
sys.path.append(os.getcwd())
from inference import plot_param_dist
from skimage.io import imsave, imread
from bonemarrow_label import BoneMarrowLabel


prediction_path = None


def print_menu():
    print('Menu:')
    print('1 - show confusion matrix')
    print('2 - insert prediction folder path')
    print('3 - calculate learning graph')
    print('4 - calculate dsc stats')
    print('5 - create dsc graphs')
    print('6 - show avg. confusion matrix')
    print('7 - create single dsc graph for predictions')
    print('8 - create density error table for predictions')
    print('q - quit')


# TODO: this is some sort of duplication from plot_param_dist, fix it some day..
def plot_params_dist(dist, param_name="Biopsies' Dice coefficient"):
    """
        Return the plot of the parameter distribution.
        Note: the parameter needs to get value from (0,1).
    """
    x_positions = np.arange(len(dist))
    width = 0.15
    background_values = [x['background'] for x in dist.values()]
    bone_values = [x['bone'] for x in dist.values()]
    fat_values = [x['fat'] for x in dist.values()]
    tissue_values = [x['tissue'] for x in dist.values()]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.bar(x_positions + 0 * width, background_values, width=width, align="center", color="black", label='Background')
    plt.bar(x_positions + 1 * width, bone_values, width=width, align="center", color="red", label='Bone')
    plt.bar(x_positions + 2 * width, fat_values, width=width, align="center", color="green", label='Fat')
    plt.bar(x_positions + 3 * width, tissue_values, width=width, align="center", color="blue", label='Other tissues')
    plt.xticks(x_positions + width, dist.keys())
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.ylim([0.0, 1.01])
    plt.xlabel(param_name, fontsize="x-large")
    plt.gca().yaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.legend(loc='lower right')
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))


def normalize_confusion_matrix(confusion_mat):
    confusion_mat = confusion_mat[:][:]
    for i in range(len(confusion_mat)):
        norm_factor = sum(confusion_mat[i])
        for j in range(len(confusion_mat[i])):
            confusion_mat[i][j] /= norm_factor
    return confusion_mat


def print_confusion_matrix_graph(confusion_mat, normalize=False, x_label_prefix='', decimal_digits=2):
    if normalize:
        confusion_mat = normalize_confusion_matrix(confusion_mat)
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[i])):
            confusion_mat[i][j] = round(confusion_mat[i][j], decimal_digits)
    labels = ['Background', 'Bone', 'Fat', 'Other Tissue']
    df_cm = pd.DataFrame(confusion_mat, index=labels, columns=labels)
    plt.figure(figsize=(7, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.ylabel('True labels')
    plt.xlabel(f'{x_label_prefix}Predicted labels')
    plt.show()


def get_confusion_matrix(path_con_mat=None, print_graph=False, normalize_mat=True):
    if path_con_mat is None:
        path_con_mat = input('Insert path to confusion matrix:\n')
    with open(path_con_mat, 'r', encoding='utf-8') as f:
        confusion_mat = json.load(f)
    if print_graph:
        print_confusion_matrix_graph(confusion_mat, normalize=normalize_mat)
    return confusion_mat


def get_avg_normalized_confusion_matrix(print_graph=False, split_train_validation=True):
    global prediction_path
    if prediction_path is None:
        prediction_path = input('Insert path to prediction folder:\n')

    train_set = []
    validation_set = []
    test_set = []

    get_names = lambda l: [x.split('.')[0] for x in l]

    if split_train_validation:
        stats_path = input('Insert path to stats file:\n')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        train_set = get_names(stats['dataset']['train'])
        validation_set = get_names(stats['dataset']['valid'])
        test_set = get_names(stats['dataset']['test'])

    name_to_confusion_matrix = {}
    for i, filename in enumerate(os.listdir(prediction_path)):
        with open(os.path.join(prediction_path, filename, f'stats - {filename}.json'), 'r', encoding='utf-8') as f:
            name_to_confusion_matrix[filename] = normalize_confusion_matrix(json.load(f))

    def show_avg_cm(matrices, prefix=''):
        avg_confusion_matrix = np.average(matrices, axis=0)
        if print_graph:
            # Normalizing just in case
            print_confusion_matrix_graph(avg_confusion_matrix, normalize=True, x_label_prefix=prefix)
        return avg_confusion_matrix

    if split_train_validation:
        show_avg_cm([v for k, v in name_to_confusion_matrix.items() if k in train_set], prefix='Train Set - ')
        show_avg_cm([v for k, v in name_to_confusion_matrix.items() if k in validation_set], prefix='Validation Set - ')
        show_avg_cm([v for k, v in name_to_confusion_matrix.items() if k in test_set], prefix='Test Set - ')
    else:
        show_avg_cm(list(name_to_confusion_matrix.values()))


def change_pred_dir():
    global prediction_path
    prediction_path = input('Insert path to prediction folder:\n')


def graph(y_values, y_label='', x_label='', show_graph=True, legend=None):
    # Note: this depends on the --validate-every cmd arg used in train, by default we set it to 10
    #       but a more generic solution is required
    x = range(10, 10*len(y_values) + 1, 10)
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

        Return: background_dsc, bone_dsc, fat_dsc, tissue_dsc
    """
    cm_path = os.path.join(prediction_path, file_name, 'stats - {}.json'.format(file_name))
    try:
        cm = get_confusion_matrix(path_con_mat=cm_path)
    except FileNotFoundError:
        print("Haven't found json file in {}".format(file_name))
        return None, None
    background_dsc = calc_dsc_from_cm(cm, BoneMarrowLabel.BACKGROUND)
    bone_dsc = calc_dsc_from_cm(cm, BoneMarrowLabel.BONE)
    fat_dsc = calc_dsc_from_cm(cm, BoneMarrowLabel.FAT)
    tissue_dsc = calc_dsc_from_cm(cm, BoneMarrowLabel.OTHER)
    return background_dsc, bone_dsc, fat_dsc, tissue_dsc


def calculate_dsc_for_all():
    """
        Format of output file: image name,bone dsc,fat dsc
    """
    global prediction_path
    if prediction_path is None:
        print('please set the path to prediction folder first.')
        return
    background_dsc_list = []
    bone_dsc_list = []
    fat_dsc_list = []
    tissue_dsc_list = []
    path_stats_file = input('Insert path to where to save the stats to.\n')
    with open(path_stats_file, 'w+') as write_f:
        for folder in os.listdir(prediction_path):
            background_dsc, bone_dsc, fat_dsc, tissue_dsc = get_dsc_from_json_file(folder)
            if bone_dsc is None:
                continue
            write_f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(folder, background_dsc, bone_dsc, fat_dsc, tissue_dsc))
            background_dsc_list.append(background_dsc)
            bone_dsc_list.append(bone_dsc)
            fat_dsc_list.append(fat_dsc)
            tissue_dsc_list.append(tissue_dsc)
        
        n = len(background_dsc_list)
        background_dsc_list.sort()
        bone_dsc_list.sort()
        fat_dsc_list.sort()
        tissue_dsc_list.sort()
        write_f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format('min', background_dsc_list[0], bone_dsc_list[0], fat_dsc_list[0], tissue_dsc_list[0]))
        write_f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format('max', background_dsc_list[-1], bone_dsc_list[-1], fat_dsc_list[-1], tissue_dsc_list[-1]))
        write_f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format('mean', sum(background_dsc_list) / n, sum(bone_dsc_list) / n, sum(fat_dsc_list) / n, sum(tissue_dsc_list) / n))


def create_dsc_graphs():
    if prediction_path is None:
        print('please set the path to prediction folder first.')
        return
    bone_dsc_dist = {}
    fat_dsc_dist = {}
    background_dsc_dist = {}
    tissue_dsc_dist = {}
    while True:
        print('Menu:')
        print('a - insert new file to add to graph')
        print('p - save graphs')
        print('q - exit')
        input_op = input()
        if input_op == 'a':
            file_name = input('insert the image name.\n')
            background_dsc, bone_dsc, fat_dsc, tissue_dsc = get_dsc_from_json_file(file_name)
            if bone_dsc is None:
                continue
            background_dsc_dist[file_name] = background_dsc
            bone_dsc_dist[file_name] = bone_dsc
            fat_dsc_dist[file_name] = fat_dsc
            tissue_dsc_dist[file_name] = tissue_dsc
            print('added {} to graph'.format(file_name))
        elif input_op == 'p':
            save_folder_path = input('Insert path to folder to save the graphs to.\n')
            os.makedirs(save_folder_path, exist_ok=True)
            dsc_background_dist_plot = plot_param_dist(background_dsc_dist)
            imsave(os.path.join(save_folder_path, 'dsc_background.png'), dsc_background_dist_plot)
            dsc_bone_dist_plot = plot_param_dist(bone_dsc_dist)
            imsave(os.path.join(save_folder_path, 'dsc_bone.png'), dsc_bone_dist_plot)
            dsc_fat_dist_plot = plot_param_dist(fat_dsc_dist)
            imsave(os.path.join(save_folder_path, 'dsc_fat.png'), dsc_fat_dist_plot)
            dsc_tissue_dist_plot = plot_param_dist(tissue_dsc_dist)
            imsave(os.path.join(save_folder_path, 'dsc_tissue.png'), dsc_tissue_dist_plot)
        elif input_op == 'q':
            return
        else:
            print('Invalid command')


def create_single_dsc_graph():
    if prediction_path is None:
        print('please set the path to prediction folder first.')
        return

    dsc = {}
    for file_name in os.listdir(prediction_path):
        background_dsc, bone_dsc, fat_dsc, tissue_dsc = get_dsc_from_json_file(file_name)
        dsc[file_name] = {
            'background': background_dsc,
            'bone': bone_dsc,
            'fat': fat_dsc,
            'tissue': tissue_dsc
        }
    save_folder_path = input('Insert path to folder to save the graphs to.\n')
    os.makedirs(save_folder_path, exist_ok=True)
    dsc_dist_plot = plot_params_dist(dsc)
    imsave(os.path.join(save_folder_path, 'dsc_all.png'), dsc_dist_plot)


def get_image_label(image):
    y_value = np.zeros(image.shape[:2])
    y_value[image[:, :, 0] != 0] = BoneMarrowLabel.BONE
    y_value[image[:, :, 1] != 0] = BoneMarrowLabel.FAT
    y_value[image[:, :, 2] != 0] = BoneMarrowLabel.OTHER
    return y_value


def create_density_error_table():
    global prediction_path
    if prediction_path is None:
        prediction_path = input('Insert path to prediction folder:\n')

    density_errors = []
    path_stats_file = input('Insert path to where to save the stats to.\n')

    with open(path_stats_file, 'w+') as f:
        for file_name in os.listdir(prediction_path):
            output_dir = os.path.join(prediction_path, file_name)
            y_pred = get_image_label(imread(os.path.join(output_dir, 'pred.png')))
            y_true = get_image_label(imread(os.path.join(output_dir, 'true.png')))

            density_error = utils.calculate_bonemarrow_density_error(y_pred, y_true)
            f.write(f'{file_name},{density_error:.4f}\n')
            density_errors.append(density_error)

        density_errors.sort()
        f.write(f'min,{density_errors[0]}\n')
        f.write(f'max,{density_errors[-1]}\n')
        f.write(f'mean,{sum(density_errors)/len(density_errors)}\n')


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
            get_avg_normalized_confusion_matrix(print_graph=True, split_train_validation=True)
        elif input_op == '7':
            create_single_dsc_graph()
        elif input_op == '8':
            create_density_error_table()
        elif input_op == 'q':
            exit()
        else:
            print('Invalid command')
