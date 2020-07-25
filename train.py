import argparse

import torch
import torch.nn as nn

from data_loader import Text_Dataset
from text_encoder import Text_Encoder as LM
import trainer



def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model',  default="basic.")
    # p.add_argument('-train', required = True)
    # p.add_argument('-valid', required = True)
    # p.add_argument('-gpu_id', type = int, default = -1)

    p.add_argument('-batch_size', type = int, default = 64)
    p.add_argument('-n_epochs', type = int, default = 20)
    p.add_argument('-print_every', type = int, default = 50)
    p.add_argument('-early_stop', type = int, default = 3)
    # p.add_argument('-iter_ratio_in_epoch', type = float, default = 1.)

    p.add_argument('-dropout', type = float, default = .5)
    p.add_argument('-embedding_dim', type = int, default = 1024)
    p.add_argument('-hidden_size', type = int, default = 1024)
    p.add_argument('-words_num', type = int, default = 15)

    p.add_argument('-n_layers', type = int, default = 1)
    p.add_argument('-max_grad_norm', type = float, default = 5.)
    p.add_argument('-lr', type = float, default = 1.)
    p.add_argument('-min_lr', type = float, default = .000001)

    config = p.parse_args()

    return config

if __name__ == '__main__':
    config = define_argparser()

    hr_dataset = Text_Dataset(data_dir='data/bird/',
                              split='train',
                              words_num=config.words_num,
                              print_shape=False)

    train_loader = torch.utils.data.DataLoader(dataset=hr_dataset,
                                            batch_size=config.batch_size,
                                            drop_last=True,
                                            shuffle=True,
                                            num_workers=0)

    hr_dataset_test = Text_Dataset(data_dir='data/bird/',
                              split='test',
                              words_num=config.words_num,
                              print_shape=False)

    test_loader = torch.utils.data.DataLoader(dataset=hr_dataset_test,
                                            batch_size=config.batch_size,
                                            drop_last=True,
                                            shuffle=True,
                                            num_workers=0)

    # loader = DataLoader(config.train, 
    #                     config.valid, 
    #                     batch_size = config.batch_size, 
    #                     device = config.gpu_id,
    #                     max_length = config.words_num
    #                     )
    
    if torch.cuda.is_available():    
        model = LM(hr_dataset.n_word, 
                    embedding_dim = config.embedding_dim, 
                    hidden_dim = config.hidden_size, 
                    n_layers = config.n_layers, 
                    dropout_p = config.dropout, 
                    max_length = config.words_num,
                    rnn_type='LSTM'
                    ).cuda()
    else:
        model = LM(hr_dataset.n_word, 
            embedding_dim = config.embedding_dim, 
            hidden_dim = config.hidden_size, 
            n_layers = config.n_layers, 
            dropout_p = config.dropout, 
            max_length = config.words_num,
            rnn_type='LSTM'
            )
    
    # Let criterion cannot count EOS as right prediction, because EOS is easy to predict.
    if torch.cuda.is_available():
        loss_weight = torch.ones(hr_dataset.n_word).cuda()
    else:
        loss_weight = torch.ones(hr_dataset.n_word)

    loss_weight[0] = 0
    criterion = nn.NLLLoss(weight = loss_weight, size_average = False)

    print(model)
    print(criterion)

    trainer.train_epoch(model, 
                        criterion, 
                        train_loader, 
                        test_loader, 
                        config
                        )