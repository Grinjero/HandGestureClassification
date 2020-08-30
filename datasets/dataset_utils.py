import os

def get_class_labels(categories_path):
    assert os.path.exists(categories_path)
    id_class_label_map = dict()
    ids = []
    labels = []
    with open(categories_path, 'r') as fd:
        for row in fd:
            row = row.rstrip()
            idx, label = row.split(' ')
            idx = int(idx)
            ids.append(idx)
            labels.append(label)

    min_id = min(ids)
    for i in range(len(ids)):
        id_class_label_map[ids[i] - min_id] = labels[i]

    return id_class_label_map
