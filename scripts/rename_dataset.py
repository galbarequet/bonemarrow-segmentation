import os

def rename_dataset():
    """
        Renames the files in the dataset to have normal number names (e.g. 1 (1) -> 1, 6b -> 6)
    """
    dataset_folder = input('Insert path to dataset folder.\n')
    sub_dirs = ['background', 'bones', 'fat', 'masks', 'raw_images']
    for dir_name in sub_dirs:
        subfolder_path = os.path.join(dataset_folder, dir_name)
        for old_name in os.listdir(subfolder_path):
            if old_name.endswith('.png'):
                # removes everything after a space (e.g. 1 (1) -> 1)
                new_name = old_name[:-4].split(' ')[0]
                
                # removes non-digits (e.g. 6b -> 6)
                new_name = ''.join(c for c in new_name if c.isdigit())

                new_name = new_name + '.png'
                # If the name is the same skip the file
                if new_name == old_name:
                    continue
                new_name_path = os.path.join(subfolder_path, new_name)
                old_name_path = os.path.join(subfolder_path, old_name)
                # If there are two files with the same name remove one (e.g. 5 (1) and 5 (2))
                if os.path.isfile(new_name_path):
                    os.remove(old_name_path)
                else:
                    os.rename(old_name_path, new_name_path)


def rename_predictions():
    """
        Renames the files in the dataset to have normal number names (e.g. 1 (1) -> 1, 6b -> 6)
    """
    predictions_folder = input('Insert path to predictions folder.\n')
    for old_name in os.listdir(predictions_folder):
        # removes everything after a space (e.g. 1 (1) -> 1)
        new_name = old_name.split(' ')[0]
        
        # removes non-digits (e.g. 6b -> 6)
        new_name = ''.join(c for c in new_name if c.isdigit())

        # If the name is the same skip the file
        if new_name == old_name:
            continue
        new_name_path = os.path.join(predictions_folder, new_name)
        old_name_path = os.path.join(predictions_folder, old_name)
        # If there are two files with the same name remove one (e.g. 5 (1) and 5 (2))
        if os.path.isdir(new_name_path):
            os.remove(old_name_path)
        else:
            os.rename(old_name_path, new_name_path)
        
        # Rename json file
        for filename in os.listdir(new_name_path):
            if filename.endswith('.json'):
                json_old_path = os.path.join(new_name_path, filename)
                json_new_path = os.path.join(new_name_path, 'stats - {}.json'.format(new_name))
                os.rename(json_old_path, json_new_path)


if __name__ == '__main__':
    while True:
        print('d - rename dataset')
        print('p - rename predictions')
        print('q - quit')
        op = input()
        if op == 'd':
            rename_dataset()
        elif op == 'p':
            rename_predictions()
        else:
            break
