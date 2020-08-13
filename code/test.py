import numpy as np
import random
#import tensorflow as tf
import os
from optparse import OptionParser
import time
import sys
import json
import tools
from collections import Counter
from collections import OrderedDict
from nn_models_pytorch import *
#from keras.models import load_model
import logging
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import tqdm

if torch.cuda.is_available():  
    print("GPU")
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)
print(device)

def gen_examples(x1, x2, match, l, y, qmask, dmask, batch_size, max_d, max_q, max_s):
    testdata = []
    for i in range(len(x1[0])):
        testdata.append(torch.Tensor(np.array([scene[i] for scene in x1])))

    for i in range(max_s):
        testdata.append(torch.Tensor(np.array([scene[i] if i < len(scene) else np.zeros((max_d, max_q)) for scene in match])))

    testdata.append(torch.Tensor(np.array(x2)))
    testdata.append(torch.Tensor(np.array(l)))
    testdata.append(torch.Tensor(np.array(qmask)))
    testdata.append(torch.Tensor((np.array(dmask))))

    mb_y = torch.Tensor(np.array(y)).type(torch.LongTensor)

    data = SampleData(testdata, mb_y)
    test = DataLoader(data, batch_size=batch_size, shuffle = True)
    return test

def pre_shuffle(x1, x2, l, y, qmask, dmask, match):
    combine = list(zip(x1, x2, l, y, qmask, dmask, match))
    np.random.shuffle(combine)
    x1, x2, l, y, qmask, dmask, match = zip(*combine)
    return list(x1), list(x2), np.array(l), list(y), np.array(qmask), np.array(dmask), list(match) 

def accuracy_score(y_pred, y_true):
    assert len(y_pred) == len(y_true)

    if len(y_true) == 0:
        return 0.

    correctly = 0

    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correctly += 1

    return float(correctly)

def eval_acc(any_model, all_examples, max_d, max_q, max_s):
    acc = 0
    n_examples = 0

    total = 0
    correct = 0
    with torch.no_grad():
        for i_batch,batch_data in tqdm.tqdm(enumerate(all_examples)):
            x = batch_data['x']
            batch = x[0].size()[0]
            for i in range(len(x)):
              x[i] = x[i].to(device)
            y = batch_data['y'].to(device)
            y_pred = any_model(x)
            predictions = torch.argmax(y_pred * x[-3], dim = 1).view(1, batch) # batch
            total += y.size(0)
            correct += (predictions == y).sum().item()
                
    return correct * 100.0 / total

class SampleData(Dataset): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y  
      
    
    def __len__(self):
        return len(self.x[0])
    
    def __getitem__(self,index):
      sample = {}
      sample_x = []
      for item in self.x:
        sample_x.append(item.type(torch.cuda.LongTensor)[index])
      sample['y'] = self.y.type(torch.cuda.LongTensor)[index]
      sample['x'] = sample_x
      return sample

def cnn_lstm_UA_DA(args):
    print('-' * 50)
    print('Load data files..')
    # get prune dictionaries 
    redundent_1, redundent_2 = tools.prune_data(args.train_file)
    # load training data
    train_examples, max_d, max_q, max_s = tools.load_jsondata(args.train_file, redundent_1, redundent_2, args.stopwords)
    # load development data
    dev_examples, a, b, c = tools.load_jsondata(args.dev_file, redundent_1, redundent_2, args.stopwords)
   
    num_train = len(train_examples[0])
    num_dev = len(dev_examples[0])
    print('-' * 50)
    print('Build dictionary..')
    word_dict = tools.build_dict(train_examples[0], train_examples[1])
    # entity dictionary for entire dataset
    entity_markers = list(set([w for w in word_dict.keys()
                              if w.startswith('@ent')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    print('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)

    print('-' * 50)
    # Load embedding file
    embeddings = tools.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
   
    (vocab_size, args.embedding_size) = embeddings.shape
    print('Building Model..')
    # build model
    if args.model_to_run == 'cnn_lstm_BiDAF':
        cnn_model = CNN_LSTM_BiDAF_Model('CNN_LSTM_BiDAF_Model', torch.FloatTensor(embeddings), num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    elif args.model_to_run == 'cnn_lstm_UA_DA':
        cnn_model = CNN_LSTM_UA_DA_Model('CNN_LSTM_UA_DA_Model', device, torch.FloatTensor(embeddings), num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    elif args.model_to_run == 'cnn_lstm_DA':
        cnn_model = CNN_LSTM_DA_Model('CNN_LSTM_DA_Model', torch.FloatTensor(embeddings), num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    elif args.model_to_run == 'cnn_lstm_UA':
        cnn_model = CNN_LSTM_UA_Model('CNN_LSTM_UA_Model', torch.FloatTensor(embeddings), num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    elif args.model_to_run == 'cnn_lstm':
        cnn_model = CNN_LSTM_Model('CNN_LSTM_Model', torch.FloatTensor(embeddings), num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    if args.pre_trained is not None:
    	cnn_model.load_state_dict(torch.load(args.pre_trained, map_location=device))
   
    cnn_model = cnn_model.to(device)
    print('Done.')

    print('-' * 50)
    print(args)

    print('-' * 50)
    print('Intial test..')
    # vectorize development data
    dev_x1, dev_x2, dev_l, dev_y, dev_qmask, dev_dmask = tools.vectorize(dev_examples, word_dict, entity_dict, max_d, max_q, max_s)
    assert len(dev_x1) == num_dev
    # pre-compute similarity matrices 
    dev_matchscore = tools.build_match(embeddings, dev_examples, word_dict, max_d, max_q, max_s)
    all_dev = gen_examples(dev_x1, dev_x2, dev_matchscore, dev_l, dev_y, dev_qmask, dev_dmask, args.batch_size, max_d, max_q, max_s)

    dev_acc = eval_acc(cnn_model, all_dev, max_d, max_q, max_s)
    print('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    if args.test_only:
        return

    torch.save(cnn_model.state_dict(), args.save_model)

    print('-' * 50)
    print('Start training..')

    # vectorize development data
    train_x1, train_x2, train_l, train_y, train_qmask, train_dmask = tools.vectorize(train_examples, word_dict, entity_dict, max_d, max_q, max_s)
    assert len(train_x1) == num_train

    train_matchscore = tools.build_match(embeddings, train_examples, word_dict, max_d, max_q, max_s)

    train_x1, train_x2, train_l, train_y, train_qmask, train_dmask, train_matchscore = pre_shuffle(train_x1, train_x2, train_l, train_y, train_qmask, train_dmask, train_matchscore) 
    start_time = time.time()
    n_updates = 0
    all_train = gen_examples(train_x1, train_x2, train_matchscore, train_l, train_y, train_qmask, train_dmask, args.batch_size, max_d, max_q, max_s)

    for epoch in tqdm.tqdm(range(args.nb_epoch)): #args.nb_epoch
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(),
                       lr=args.learning_rate)
        running_loss = 0.0

        for i_batch,batch_data in enumerate(all_train):
            x = batch_data['x']
            for i in range(len(x)):
              x[i] = x[i].to(device)
            y = batch_data['y'].to(device)
            y_pred = cnn_model(x)
            batch = y.size()[0]
            loss = criterion(y_pred.float(), y.view(batch))
            running_loss = loss.item()
            optimizer.zero_grad()  
            loss = loss.backward()  
            optimizer.step()

            print('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, i_batch, len(train_x1), running_loss, time.time() - start_time))
            n_updates = n_updates+1 
            if n_updates % 100 == 0:
                samples = sorted(np.random.choice(num_train, min(num_train, num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                	[train_x2[k] for k in samples],
                    [train_matchscore[k] for k in samples],
                    train_l[samples],
                    [train_y[k] for k in samples],
                    train_qmask[samples],
                    train_dmask[samples],
                    args.batch_size,max_d, max_q, max_s)
                print('Train accuracy: %.2f %%' % eval_acc(cnn_model, sample_train, max_d, max_q, max_s))
                dev_acc = eval_acc(cnn_model, all_dev, max_d, max_q, max_s)
                print('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    print('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    torch.save(cnn_model.state_dict(), args.save_model)



    
    #train = []
    #for i in range(25):
	#    train.append(torch.rand(32,92))
    #for i in range(25):
	#    train.append(torch.rand(32,92,35))
    #train.append(torch.rand(32, 35))
    #train.append(torch.rand(32, 17))
    #train.append(torch.rand(32, 35))
    #train.append(torch.rand(32, 25))
    #y = torch.rand(32, 1)
    #data = SampleData(train, y)
    #train = DataLoader(data, batch_size=2)

    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(cnn_model.parameters(),
    #                   lr=args.learning_rate)
    #for i_batch,batch_data in enumerate(train):
    #    x = batch_data['x']
    #    y = batch_data['y']
    #    y_pred = cnn_model(x)
    #    loss = criterion(y_pred.float(), y.view(2))
    #    optimizer.zero_grad()  
    #    loss.backward()  
    #    optimizer.step()


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_dict = ['cnn_lstm_UA', 'cnn_lstm_UA_DA', 'cnn_lstm_BiDAF', 'cnn_lstm_DA', 'cnn_lstm']
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--logging_to_file',
                      action='store',
                      dest='logging_to_file',
                      default=None,
                      help='logging to a file or stdout')
    parser.add_option('--model',
                      action='store',
                      dest='model_to_run',
                      default=None,
                      help='model to train (available: %s)' % ', '.join(model_dict))
    parser.add_option('--nb_epoch',
                      action='store',
                      dest='nb_epoch',
                      default=100,
                      help='number of epochs to train the model with')
    parser.add_option('--embedding_size',
                      action='store',
                      dest='embedding_size',
                      default=100,
                      help='embedding size of the inputs')
    parser.add_option('--train_file',
                      action='store',
                      dest='train_file',
                      default=None,
                      help='train file')
    parser.add_option('--dev_file',
                      action='store',
                      dest='dev_file',
                      default=None,
                      help='dev file')
    parser.add_option('--save_model',
                      action='store',
                      dest='save_model',
                      default=None,
                      help='model to save')
    parser.add_option('--random_seed',
                      action='store',
                      dest='random_seed',
                      default=1234,
                      help='random seed')
    parser.add_option('--embedding_file',
                      action='store',
                      dest='embedding_file',
                      default=None,
                      help='embedding file')
    parser.add_option('--stopwords',
                      action='store',
                      dest='stopwords',
                      default=None,
                      help='stopwords')
    parser.add_option('--test_only',
                      action='store',
                      dest='test_only',
                      default=False,
                      help='If just to test the model')
    parser.add_option('--pre_trained',
                      action='store',
                      dest='pre_trained',
                      default=None,
                      help='pre-trained model')
    parser.add_option('--batch_size',
                      action='store',
                      dest='batch_size',
                      default=4,
                      help='training and testing batch size')
    parser.add_option('--hidden_size',
                      action='store',
                      dest='hidden_size',
                      default=16,
                      help='hidden size of LSTM')
    parser.add_option('--query_filters',
                      action='store',
                      dest='query_filters',
                      default=50,
                      help='number of filters for query CNN')
    parser.add_option('--utterance_filters',
                      action='store',
                      dest='utterance_filters',
                      default=50,
                      help='number of filters for utterance CNN')
    parser.add_option('--dropout',
                      action='store',
                      dest='dropout',
                      default=0.2,
                      help='dropout rate for LSTM')
    parser.add_option('--learning_rate',
                      action='store',
                      dest='learning_rate',
                      default=0.001,
                      help='learning rate of the model')
    (options, args) = parser.parse_args()

    fixed_seed_num = int(options.random_seed)
    options.nb_epoch = int(options.nb_epoch)
    options.batch_size = int(options.batch_size)
    options.embedding_size = int(options.embedding_size)
    options.hidden_size = int(options.hidden_size)
    options.query_filters = int(options.query_filters)
    options.utterance_filters = int(options.utterance_filters)
    options.dropout = float(options.dropout)
    options.learning_rate = float(options.learning_rate)

    np.random.seed(fixed_seed_num)
    random.seed(fixed_seed_num)
    #tf.set_random_seed(fixed_seed_num)
    if options.train_file is None:
        raise ValueError('train_file is not specified.')
    if options.dev_file is None:
        raise ValueError('dev_file is not specified.')
    if options.stopwords is None:
        raise ValueError('stopwords are not specified.')
    if options.embedding_file is not None:
        dim = tools.get_dim(options.embedding_file)
        if (options.embedding_size is not None) and (options.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                                (options.embedding_size, options.embedding_file, dim))
    else:
        raise ValueError('embedding_file is not specified.')

    FORMAT = '[%(levelname)-8s] [%(asctime)s] [%(name)-15s]: %(message)s'
    DATEFORMAT = '%Y-%m-%d %H:%M:%S'

    if options.logging_to_file:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT,
                            filename=options.logging_to_file)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT)

    logger_exp = logging.getLogger('experiments')
   
    if not options.logging_to_file:
        logger_exp.info('logging to stdout')

    if options.model_to_run not in model_dict:
        raise Exception('model `%s` not implemented' % options.model_to_run)

    cnn_lstm_UA_DA(options)

