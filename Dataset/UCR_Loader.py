import os
import numpy as np
from sklearn import model_selection
import logging

logger = logging.getLogger(__name__)


def load_UCR_dataset(file_name, folder_path):

    train_path = folder_path + '/' + file_name + '/' + file_name + "_TRAIN.tsv"
    test_path = folder_path + '/' + file_name + '/' + file_name + "_TEST.tsv"

    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest


def split_dataset(data, label, validation_ratio):
    # splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    # _, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    # val_data = data[val_indices]
    # val_label = label[val_indices]
    # return val_data, val_label
    _, counts = np.unique(label, return_counts=True)
    if np.any(counts < 2):
        logger.warning("Some classes <2 samples; doing a RANDOM split for val set")
        splitter = model_selection.ShuffleSplit(
            n_splits=1,
            test_size=validation_ratio,
            random_state=1234
        )
        # ShuffleSplit.split yields (train_idx, val_idx)
        _, val_indices = next(splitter.split(X=np.arange(len(label))))
    else:
        splitter = model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_ratio,
            random_state=1234
        )
        _, val_indices = next(splitter.split(X=np.zeros(len(label)), y=label))

    val_data = data[val_indices]
    val_label = label[val_indices]
    return val_data, val_label


for problem in os.listdir('UCR'):
    if problem.startswith('.') or not os.path.isdir(os.path.join('UCR', problem)):
        continue
    problem_dir = os.path.join('UCR', problem)
    train_path = os.path.join(problem_dir, f"{problem}_TRAIN.tsv")
    test_path  = os.path.join(problem_dir, f"{problem}_TEST.tsv")

    # ← skip hidden entries, non‐dirs or missing TRAIN/TEST
    if problem.startswith('.') \
       or not os.path.isdir(problem_dir) \
       or not os.path.isfile(train_path) \
       or not os.path.isfile(test_path):
        continue
    print(problem)
    Data = {}
    X_train, y_train, X_test, y_test = load_UCR_dataset(file_name=problem, folder_path=os.getcwd()+'/UCR')
    if X_train is None or y_train is None:
        continue
    
    # uniq, counts = np.unique(y_train, return_counts=True)
    # if np.any(counts < 2):
    #     print(f"[{args.dataset}] WARNING: some classes <2 samples; doing a random (non-stratified) split")
    #     stratify_arg = None
    # else:
    #     stratify_arg = y_train
    # # train → train/val
    # X_tr, X_val, y_tr, y_val = train_test_split(
    #     X_train, y_train,
    #     test_size=args.val_size,
    #     random_state=args.seed,
    #     stratify=stratify_arg
    # )
    
    
    
    X_val, Data['val_label'] = split_dataset(X_train, y_train, 0.5)
    Data['val_data'] = np.expand_dims(X_val, axis=1)
    max_seq_len = X_train.shape[1]
    Data['max_len'] = max_seq_len
    Data['train_data'] = np.expand_dims(X_train, axis=1)
    Data['train_label'] = y_train
    Data['All_train_data'] = np.expand_dims(X_train, axis=1)
    Data['All_train_label'] = y_train

    Data['test_data'] = np.expand_dims(X_test, axis=1)
    Data['test_label'] = y_test
    np.save(os.getcwd()+'/UCR/' + problem + "/" + problem, Data, allow_pickle=True)
    logger.info("{} samples will be used for training".format(len(Data['train_label'])))


