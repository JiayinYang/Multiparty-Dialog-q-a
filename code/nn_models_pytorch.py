import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_UA_DA_Model(nn.Module):
    def __init__(self, name, device, embeddings, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        super(CNN_LSTM_UA_DA_Model, self).__init__()
        self.device = device
        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        # embedding layers
        self.embedding_layer_utterance = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer_query = nn.Embedding.from_pretrained(embeddings)
        
        # query convolution layer
        self.conv_embedding_query = nn.Conv2d(1, self.nb_filters_query, kernel_size = (1, self.embedding_size))
        self.relu = nn.ReLU()
        
        # utterance embedding layer
        self.conv_embedding_utterances = []
        self.pool = []
        for j in range(2, 6):
            self.conv_embedding_utterances.append(nn.Conv2d(2, self.nb_filters_utterance, kernel_size = (j, self.embedding_size)))
            self.pool.append(nn.MaxPool2d((self.nb_utterance_token-j+1, 1)))
        for j in range(len(self.conv_embedding_utterances)):
          self.conv_embedding_utterances[j] = self.conv_embedding_utterances[j].to(self.device)
          self.pool[j] = self.pool[j].to(self.device)
        self.conv_reshape_scene = nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (1, self.nb_filters_utterance*4))
        
        # context embedding layers for both dialog and query 
        self.lstm_scene = nn.Sequential(nn.LSTM(self.nb_filters_utterance*4, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        self.lstm_query = nn.Sequential(nn.LSTM(self.embedding_size, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        
        # output layer
        self.dense = nn.Linear(self.embedding_size + self.nb_hidden_unit*4, self.nb_classes)
        self.softmax = nn.Softmax() 
        

    def forward(self, x):
        # utternace level attention matrix
        #print(self.device)
        self.batch = x[0].size()[0]
        attn = DocAttentionMap([self.nb_utterance_token, self.embedding_size],[92,35], self.device)
        
        # 3-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            #print(self.embedding_layer_utterance(x[i]).size())
            embedding_utter = self.embedding_layer_utterance(x[i]).view(self.batch,1, self.nb_utterance_token, self.embedding_size)
            size = x[i+self.nb_utterances].size()
            attentin_array = x[i+self.nb_utterances].view(size[0]*size[1], size[2])
            attention = attn(attentin_array.type(torch.FloatTensor))
            doc_att_map = attention.view(size[0], 1, size[1], attention.size()[1])
            embedding_utterances.append(torch.cat([embedding_utter, doc_att_map], dim = 1))
        
        # convolution embedding input for query
        #print("herer")
        conv_embedding_query = self.embedding_layer_utterance(x[-4]).view(self.batch, 1, self.nb_query_token, self.embedding_size)
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(x[-4]) 
        # convolution output for query
        conv_q = self.conv_embedding_query(conv_embedding_query) 
        conv_q = self.relu(conv_q)
        conv_q = conv_q.view(self.batch, self.nb_query_token,self.nb_filters_query) 

        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = self.relu(self.conv_embedding_utterances[j-2](embedding_utterances[i]))
                pool_u = self.pool[j - 2](conv_u) 
                utter.append(pool_u)
            scene.append(torch.cat(utter, dim = 3).view(self.batch, self.nb_filters_utterance*4,1))

        # dialog matrix
        scene = torch.cat(scene, dim = 2).permute(0, 2, 1)
        # convolution output of dialog matrix 
        reshape_scene = scene.view(self.batch, 1, self.nb_utterances, self.nb_filters_utterance*4)
        single = self.relu(self.conv_reshape_scene(reshape_scene))
        single = single.view(self.batch, self.nb_utterances, self.nb_filters_utterance)

        # context embedding for both dialog and query
        bi_d_rnn, _ = self.lstm_scene(scene)
        bi_d_rnn = bi_d_rnn[:,-1,:] 
        bi_q_rnn, _ = self.lstm_query(embedding_query)
        bi_q_rnn = bi_q_rnn[:,-1,:]
        
        # dialog level attention vector
        cross_attention = crossatt()
        att_vector = cross_attention([single, conv_q, x[-1], x[-2]])

        # output
        merged_vectors = torch.cat([bi_d_rnn, bi_q_rnn, att_vector], dim = 1)
        classes = self.softmax(self.dense(merged_vectors))
        
        # masking 
        mask = masking_lambda(self.nb_classes)
        normalized_classes = mask([classes, x[-3]])
        return normalized_classes


class CNN_LSTM_UA_Model(nn.Module):
    def __init__(self, name, embeddings, batch, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        super(CNN_LSTM_UA_DA_Model, self).__init__()
        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch = batch
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        # embedding layers
        self.embedding_layer_utterance = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer_query = nn.Embedding.from_pretrained(embeddings)
        
        # utterance embedding layers      
        self.conv_embedding_utterances = []
        self.pool = []
        self.relu = nn.ReLU()
        for j in range(2, 6):
            self.conv_embedding_utterances.append(nn.Conv2d(2, self.nb_filters_utterance, kernel_size = (j, self.embedding_size)))
            self.pool.append(nn.MaxPool2d((self.nb_utterance_token-j+1, 1)))
        self.conv_reshape_scene = nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (1, self.nb_filters_utterance*4))
        
        # context embedding layers for both dialog and query 
        self.lstm_scene = nn.Sequential(nn.LSTM(self.nb_filters_utterance*4, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        self.lstm_query = nn.Sequential(nn.LSTM(self.embedding_size, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        
        # output layer
        self.dense = nn.Linear(self.nb_hidden_unit*4, self.nb_classes)
        self.softmax = nn.Softmax() #dimention
        

    def forward(self, x):
        # utternace level attention matrix
        attn = DocAttentionMap([self.nb_utterance_token, self.embedding_size],[92,35])
        embedding_utterances = []

        # 3-D embedding for utterances
        for i in range(self.nb_utterances):
            embedding_utter = self.embedding_layer_utterance(x[i]).view(self.batch,1, self.nb_utterance_token, self.embedding_size)
            size = x[i+self.nb_utterances].size()
            attentin_array = x[i+self.nb_utterances].view(size[0]*size[1], size[2])
            attention = attn(attentin_array.type(torch.FloatTensor))
            doc_att_map = attention.view(size[0],1, size[1],attention.size()[1])
            embedding_utterances.append(torch.cat([embedding_utter, doc_att_map], dim = 1))
       
        # lstm embedding input for query
        embedding_query = self.embedding_layer_query(x[-4]) 
         
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = self.relu(self.conv_embedding_utterances[j-2](embedding_utterances[i]))
                pool_u = self.pool[j - 2](conv_u)
                utter.append(pool_u)
            scene.append(torch.cat(utter, dim = 3).view(self.batch, self.nb_filters_utterance*4,1)) 

        # dialog matrix
        scene = torch.cat(scene, dim = 2).permute(0, 2, 1)

        # context embedding for both dialog and query
        bi_d_rnn, _ = self.lstm_scene(scene)
        bi_d_rnn = bi_d_rnn[:,-1,:]
        bi_q_rnn, _ = self.lstm_query(embedding_query)
        bi_q_rnn = bi_q_rnn[:,-1,:]

        merged_vectors = torch.cat([bi_d_rnn, bi_q_rnn], dim = 1)
        classes = self.softmax(self.dense(merged_vectors))
        
        # masking
        mask = masking_lambda(self.nb_classes)
        normalized_classes = mask([classes, x[-3]])

        return normalized_classes

class CNN_LSTM_DA_Model(nn.Module):
    def __init__(self, name, embeddings, batch, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        super(CNN_LSTM_DA_Model, self).__init__()
        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch = batch
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        # embedding layers
        self.embedding_layer_utterance = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer_query = nn.Embedding.from_pretrained(embeddings)
        
        # query convolution layer
        self.conv_embedding_query = nn.Conv2d(1, self.nb_filters_query, kernel_size = (1, self.embedding_size))
        self.relu = nn.ReLU()
        
        # utterance embedding layer
        self.conv_embedding_utterances = []
        self.pool = []
        for j in range(2, 6):
            self.conv_embedding_utterances.append(nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (j, self.embedding_size)))
            self.pool.append(nn.MaxPool2d((self.nb_utterance_token-j+1, 1)))
        self.conv_reshape_scene = nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (1, self.nb_filters_utterance*4))
        
        # context embedding layers for both dialog and query 
        self.lstm_scene = nn.Sequential(nn.LSTM(self.nb_filters_utterance*4, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        self.lstm_query = nn.Sequential(nn.LSTM(self.embedding_size, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        
        # output layer
        self.dense = nn.Linear(self.embedding_size + self.nb_hidden_unit*4, self.nb_classes)
        self.softmax = nn.Softmax() 
        

    def forward(self, x):
        # 2-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = self.embedding_layer_utterance(x[i]).view(self.batch,1, self.nb_utterance_token, self.embedding_size)
            embedding_utterances.append(embedding_utter)
        
        # convolution embedding input for query
        conv_embedding_query = self.embedding_layer_utterance(x[-4]).view(self.batch, 1, self.nb_query_token, self.embedding_size)
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(x[-4]) 
        # convolution output for query
        conv_q = self.conv_embedding_query(conv_embedding_query) 
        conv_q = self.relu(conv_q)
        conv_q = conv_q.view(self.batch, self.nb_query_token,self.nb_filters_query) 

        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = self.relu(self.conv_embedding_utterances[j-2](embedding_utterances[i]))
                pool_u = self.pool[j - 2](conv_u) 
                utter.append(pool_u)
            scene.append(torch.cat(utter, dim = 3).view(self.batch, self.nb_filters_utterance*4,1))

        # dialog matrix
        scene = torch.cat(scene, dim = 2).permute(0, 2, 1)
        # convolution output of dialog matrix 
        reshape_scene = scene.view(self.batch, 1, self.nb_utterances, self.nb_filters_utterance*4)
        single = self.relu(self.conv_reshape_scene(reshape_scene))
        single = single.view(self.batch, self.nb_utterances, self.nb_filters_utterance)

        # context embedding for both dialog and query
        bi_d_rnn, _ = self.lstm_scene(scene)
        bi_d_rnn = bi_d_rnn[:,-1,:] 
        bi_q_rnn, _ = self.lstm_query(embedding_query)
        bi_q_rnn = bi_q_rnn[:,-1,:]
        
        # dialog level attention vector
        cross_attention = crossatt()
        att_vector = cross_attention([single, conv_q, x[-1], x[-2]])

        # output
        merged_vectors = torch.cat([bi_d_rnn, bi_q_rnn, att_vector], dim = 1)
        classes = self.softmax(self.dense(merged_vectors))
        
        # masking 
        mask = masking_lambda(self.nb_classes)
        normalized_classes = mask([classes, x[-3]])
        return normalized_classes

class CNN_LSTM_Model(nn.Module):
    def __init__(self, name, device, embeddings, batch, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        super(CNN_LSTM_Model, self).__init__()
        self.device = device
        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch = batch
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        # embedding layers
        self.embedding_layer_utterance = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer_query = nn.Embedding.from_pretrained(embeddings)

        # utterance embedding layers
        self.relu = nn.ReLU()
        self.conv_embedding_utterances = []
        self.pool = []
        for j in range(2, 6):
            self.conv_embedding_utterances.append(nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (j, self.embedding_size)))
            self.pool.append(nn.MaxPool2d((self.nb_utterance_token-j+1, 1)))
        for j in range(len(self.conv_embedding_utterances)):
          self.conv_embedding_utterances[j] = self.conv_embedding_utterances[j].to(self.device)
          self.pool[j] = self.pool[j].to(self.device)
        
        # contextual embedding layers
        self.lstm_scene = nn.Sequential(nn.LSTM(self.nb_filters_utterance*4, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        self.lstm_query = nn.Sequential(nn.LSTM(self.embedding_size, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True))
        
        # output layer
        self.dense = nn.Linear(self.nb_hidden_unit*4, self.nb_classes)
        self.softmax = nn.Softmax() 
        

    def forward(self, x):
        # 2-D embedding for utterances
        embedding_utterances = []

        for i in range(self.nb_utterances):
            embedding_utter = self.embedding_layer_utterance(x[i])
            embedding_utter = embedding_utter.view(self.batch,1, self.nb_utterance_token, self.embedding_size)
            embedding_utterances.append(embedding_utter)
        
        # convolution embedding input for query
        embedding_query = self.embedding_layer_query(x[-4]) 

        # utterance embeddings
        scene = []

        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = self.relu(self.conv_embedding_utterances[j-2](embedding_utterances[i]))
                pool_u = self.pool[j - 2](conv_u)
                utter.append(pool_u)
            scene.append(torch.cat(utter, dim = 3).view(self.batch, self.nb_filters_utterance*4,1))  

        # dialog matrix
        scene = torch.cat(scene, dim = 2).permute(0, 2, 1)

        # context embedding for both dialog and query
        bi_d_rnn, _ = self.lstm_scene(scene)
        bi_d_rnn = bi_d_rnn[:,-1,:]
        bi_q_rnn, _ = self.lstm_query(embedding_query)
        bi_q_rnn = bi_q_rnn[:,-1,:]

        # output
        merged_vectors = torch.cat([bi_d_rnn, bi_q_rnn], dim = 1) 
        classes = self.softmax(self.dense(merged_vectors))
        
        # masking
        mask = masking_lambda(self.nb_classes)
        normalized_classes = mask([classes, x[-3]])

        return normalized_classes

class CNN_LSTM_BiDAF_Model(nn.Module):
    def __init__(self, name, embeddings, batch, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=75):

        super(CNN_LSTM_BiDAF_Model, self).__init__()
        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch = batch
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = 75
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        # character embedding layers
        self.embedding_layer_utterance_char = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=1)
        nn.init.uniform_(self.embedding_layer_utterance_char.weight, -0.001, 0.001)

        self.embedding_layer_query_char = nn.Embedding(self.vocabulary_size, self.embedding_size)
        nn.init.uniform_(self.embedding_layer_query_char.weight, -0.001, 0.001)

        # word embedding layers
        self.embedding_layer_utterance = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer_query = nn.Embedding.from_pretrained(embeddings)
        
        # highway network
        assert self.nb_hidden_unit * 2 == (self.embedding_size + self.nb_filters_utterance)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(self.nb_hidden_unit * 2, self.nb_hidden_unit * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(self.nb_hidden_unit * 2, self.nb_hidden_unit * 2),
                                  nn.Sigmoid()))

        # character embedding convolution layer
        self.conv_embedding_query_char = nn.Conv2d(1, self.nb_filters_query, kernel_size = (1, self.embedding_size))
        self.conv_embedding_utterances_char = nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (1, self.embedding_size))
        self.relu = nn.ReLU()
        
        # utterance embeddings
        self.conv_embedding_utterances = []
        self.pool = []
        for j in range(2, 6):
            self.conv_embedding_utterances.append(nn.Conv2d(1, self.nb_filters_utterance, kernel_size = (j, self.embedding_size + self.nb_filters_utterance)))
            self.pool.append(nn.MaxPool2d((self.nb_utterance_token-j+1, 1)))
        
        # contexutal embedding layer
        self.lstm_scene = nn.LSTM(self.nb_hidden_unit * 2, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True)
        self.lstm_query = nn.LSTM(self.nb_hidden_unit * 2, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True)
        self.softmax = nn.Softmax() #dimention

        # attention flow layer
        self.att_weight_c = nn.Linear(self.nb_hidden_unit * 2, 1)
        self.att_weight_q = nn.Linear(self.nb_hidden_unit * 2, 1)
        self.att_weight_cq = nn.Linear(self.nb_hidden_unit * 2, 1)
        
        # modeling layer
        self.lstm1 = nn.LSTM(self.nb_hidden_unit * 8, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True)
        self.lstm2 = nn.LSTM(self.nb_hidden_unit * 2, hidden_size = self.nb_hidden_unit, dropout = self.dropout, bidirectional=True, batch_first = True)

        # output layer
        self.p1_weight_g = nn.Linear(self.nb_hidden_unit * 8, 1)
        self.p1_weight_m = nn.Linear(self.nb_hidden_unit * 2, 1)
        self.p2_weight_g = nn.Linear(self.nb_hidden_unit * 8, 1)
        self.p2_weight_m = nn.Linear(self.nb_hidden_unit * 2, 1)

        self.output_LSTM = nn.LSTM(input_size=self.nb_hidden_unit * 2,
                                hidden_size=self.nb_hidden_unit,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.dropout)
        self.dense = nn.Linear(self.nb_utterances*2, self.nb_classes)


    def att_flow_layer(self, c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

    def highway_network(self, x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            
            # (batch, seq_len, hidden_size * 2)
            return x

    def forward(self, x):
        # character embedding for query
        embedding_query_char = self.embedding_layer_query_char(x[-4]) 
        conv_embedding_query = embedding_query_char.view(self.batch, 1, self.nb_query_token, self.embedding_size)
        conv_q = self.relu(self.conv_embedding_query_char(conv_embedding_query)).view(self.batch, self.nb_query_token, self.nb_filters_query)
        # word embedding for query
        embedding_query = self.embedding_layer_query(x[-4]) 
        embedding_query = embedding_query.view(self.batch, self.nb_query_token, self.embedding_size)
        
        # highway network
        query = self.highway_network(embedding_query, conv_q)

        scene = []
        for i in range(self.nb_utterances):
            # character embedding for utterances
            embedding_utter_char = self.embedding_layer_utterance_char(x[i]).view(self.batch, 1, self.nb_utterance_token, self.embedding_size)
            embedding_utter_char = self.relu(self.conv_embedding_utterances_char(embedding_utter_char)).view(self.batch, self.nb_utterance_token, self.nb_filters_utterance)
            # word embedding for utterances
            embedding_utter = self.embedding_layer_utterance(x[i]).view(self.batch, self.nb_utterance_token, self.embedding_size)
            # highway network
            scene_embedding = self.highway_network(embedding_utter, embedding_utter_char).view(self.batch, 1, self.nb_utterance_token, self.nb_hidden_unit*2)
            scene.append(scene_embedding)

        # utterance embeddings
        utterances = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 5):
                conv_u = self.relu(self.conv_embedding_utterances[j-2](scene[i])) 
                pool_u = self.pool[j - 2](conv_u) 
                utter.append(pool_u)
            utterances.append(torch.cat(utter, dim = 3).view(self.batch, self.nb_filters_utterance*3,1))
        # dialog matrix
        utterances = torch.cat(utterances, dim = 2).permute(0, 2, 1)
        
        # contextual embedding
        bi_d_rnn, _ = self.lstm_scene(utterances)
        bi_q_rnn, _ = self.lstm_query(query)
        
        # attention flow layer
        attention_flow_layer = self.att_flow_layer(bi_d_rnn, bi_q_rnn)
        
        # lstm modeling
        modeling_layer = self.lstm2(self.lstm1(attention_flow_layer)[0])[0]

        # output layer
        p1 = (self.p1_weight_g(attention_flow_layer) + self.p1_weight_m(modeling_layer)).squeeze()   
        m2 = self.output_LSTM(modeling_layer)[0]
        p2 = (self.p2_weight_g(attention_flow_layer) + self.p2_weight_m(m2)).squeeze()        
        p = torch.cat([p1, p2], dim = -1)
        classes = self.softmax(self.dense(p))
        
        # masking 
        mask = masking_lambda(self.nb_classes)
        normalized_classes = mask([classes, x[-3]])

        return normalized_classes


class masking_lambda(nn.Module):
    def __init__(self, nb_classes):
        super(masking_lambda, self).__init__()
        self.nb_classes = nb_classes

    def forward(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = torch.sum(masked_classes, 1)
        masked_sum_ = torch.unsqueeze(masked_sum_, -1)
        masked_sum = torch.repeat_interleave(masked_sum_, self.nb_classes, 1)
        masked_classes = masked_classes / masked_sum
        masked_classes = torch.clamp(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

class crossatt(nn.Module):
    def __init__(self):
        super(crossatt, self).__init__()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        doc, query, doc_mask, q_mask = x[0], x[1], x[2], x[3]
        trans_doc = doc.permute(0,2,1)
        match_score = self.tanh( torch.matmul(query, trans_doc))
        query_to_doc_att = self.softmax(torch.sum(match_score, 1))
        doc_to_query_att = self.softmax(torch.sum(match_score, -1))

        alpha = query_to_doc_att*doc_mask
        a_sum = torch.sum(alpha, 1)
        _a_sum = torch.unsqueeze(a_sum, -1)
        alpha = alpha/_a_sum

        beta = doc_to_query_att*q_mask
        b_sum = torch.sum(beta, 1)
        _b_sum = torch.unsqueeze(b_sum, 1)
        beta = beta/_b_sum

        doc_vector = torch.matmul(trans_doc, alpha.view(alpha.size()[0], alpha.size()[1], 1))
        trans_que = query.permute(0,2,1)
        que_vector = torch.matmul(trans_que, beta.view(beta.size()[0], beta.size()[1], 1))
        doc_vector = doc_vector.view(doc_vector.size()[0],doc_vector.size()[1])
        que_vector = que_vector.view(que_vector.size()[0],que_vector.size()[1])
        final_hidden = torch.cat([doc_vector, que_vector], dim = 1) # [2, 100]
        return final_hidden


class DocAttentionMap(nn.Module):
    def __init__(self, output_dim, input_shape, device, **kwargs):
        self.output_dim = output_dim
        self.device = device
        super(DocAttentionMap, self).__init__(**kwargs)
        sampler = torch.distributions.Uniform(low=-1, high=1)
        self.U = sampler.sample((input_shape[-1], self.output_dim[1]))

    def forward(self, x):
        self.U = self.U.to(self.device)
        x = x.to(self.device)
        dotproduct = torch.mm(x, self.U)
        return torch.tanh(dotproduct)