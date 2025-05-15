import os
from utils import args
from Dataset import dataloader
from models.runner import supervised, pre_training, linear_probing


if __name__ == '__main__':
    config = args.Initialization(args)

for problem in os.listdir(config['data_dir']):
    if config.get('problem') and problem != config['problem']:
        continue
    config['problem'] = problem
    print(problem)
    Data = dataloader.data_loader(config)
    #print(Data.keys())

    # print(config['Training_mode'])
    # print(config['Model_Type'])
    if config['Training_mode'] == 'Pre_Training':
        if config['Model_Type'] == 'Series2Vec':
            #best_aggr_metrics_test, all_metrics = pre_training(config, Data)
            print("been here")
            train_repr, train_labels, test_repr, test_labels = pre_training(config, Data)
    elif config['Training_mode'] == 'Linear_Probing':
        best_aggr_metrics_test, all_metrics = linear_probing(config, Data)
    elif config['Training_mode'] == 'Supervised':
        best_aggr_metrics_test, all_metrics = supervised(config, Data)

    print("representations:")
    print("train set and label shape:", train_repr.shape, train_labels.shape)
    print("test set and label shape:", test_repr.shape, test_labels.shape)
    print(train_repr[0])
    #print(all_metrics)
    # print(best_aggr_metrics_test)
    # print_str = 'Best Model Test Summary: '
    # for k, v in best_aggr_metrics_test.items():
    #     print_str += '{}: {} | '.format(k, v)
    # print(print_str)

    # with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
    #     for k, v in all_metrics.items():
    #         file.write(f'{k}: {v}\n')


    """
python main.py \
  --data_dir ./UCR \
  --output_dir ./outputs/pretrain \
  --Training_mode Pre_Training \
  --Model_Type Series2Vec
  
    """