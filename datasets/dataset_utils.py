import os

def get_class_labels(categories_path):
    assert os.path.exists(categories_path)
    id_class_label_map = dict()

    with open(categories_path, 'r') as fd:
        for row in fd:
            idx, label = row.split(' ')
            idx = int(idx)

            id_class_label_map[idx] = str.rstrip(label)

    return id_class_label_map
