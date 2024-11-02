import os
import dgl
import torch
import torch.nn as nn
from torch import optim
import tqdm
from dgl.nn.pytorch import GraphConv,GATConv
from transformers import BertModel, BertTokenizer, BertConfig
from dgl.data.utils import load_graphs, load_info
from sklearn.metrics import classification_report, accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42
torch.manual_seed(seed)

class Dataset_load:
    def __init__(self,usedumpData=True,dumpFilename=" "):
        '''
        :param usedumpData:
        :param dumpFilename:
        '''

        if usedumpData==True and os.path.exists(dumpFilename):
            GS, _ = load_graphs(dumpFilename)
            data = load_info(dumpFilename + "_info.pkl")
            self.labelName = data['labelName']
            self.labelNameSet=data['labelNameSet']
            self.graphs = GS
            self.labelId = data['labelId']
            self.train_index = data['train_index']
            self.test_index = data['test_index']
            self.valid_index = data['valid_index']
            info ='Successfully Load dump data from {0}'.format(dumpFilename)
            print(info)

        self.train_watch = 0
        self.test_watch =  0
        self.valid_watch = 0
        self.epoch_over = False

    def __next_batch(self,name,batch_size):
        graphs =[]
        labels =[]

        for i in range(batch_size):
            if name == 'train':
                graphs.append(self.graphs[self.train_index[self.train_watch]])
                labels.append(self.labelId[self.train_index[self.train_watch]])

                if (self.train_watch + 1) == len(self.train_index):
                    self.epoch_over +=1
                self.train_watch = (self.train_watch + 1) % len(self.train_index)
            elif name =='valid':
                graphs.append(self.graphs[self.valid_index[self.valid_watch]])
                labels.append(self.labelId[self.valid_index[self.valid_watch]])
                self.valid_watch = (self.valid_watch + 1) % len(self.valid_index)
            else:
                graphs.append(self.graphs[self.test_index[self.test_watch]])
                labels.append(self.labelId[self.test_index[self.test_watch]])
                self.test_watch = (self.test_watch + 1) % len(self.test_index)
        return dgl.batch(graphs),torch.tensor(labels)
    def next_train_batch(self,batch_size):
        return self.__next_batch('train',batch_size)
    def next_valid_batch(self,batch_size):
        return self.__next_batch('valid',batch_size)
    def next_test_batch(self,batch_size):
        return self.__next_batch('test',batch_size)

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        config = BertConfig.from_pretrained("bert-mini", dropout= 0.5)
        self.bert = BertModel.from_pretrained("bert-mini", config=config)
        self.fc1 = nn.Linear(in_features=25, out_features=256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = x.unsqueeze(1).expand(-1, 25, -1)
        hidden_rep, cls_head = self.bert(inputs_embeds=x, return_dict=False)
        return cls_head

class Feature_extractor(nn.Module):
    def __init__(self,feature_width=100,nb_classes=None):
        self.device = device
        super(Feature_extractor,self).__init__()
        filter_num = ['None',32,64,128,256,512]
        kernel_size = ['None',8,8,8,8,8]
        conv_stride_size = ['None',1,1,1,1,1]
        pool_stride_size = ['None',4,4,4,4,4]
        pool_size = ['None',8,8,8,8,8]

        self._1Conv1D = nn.Conv1d(in_channels=1, out_channels=filter_num[1], stride=conv_stride_size[1], kernel_size=kernel_size[1],   padding=kernel_size[1]//2)
        self._2BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._3ReLU = nn.ReLU()
        self._4Conv1D = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], stride=conv_stride_size[1], kernel_size = kernel_size[1], padding=kernel_size[1]//2)
        self._5BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._6ReLU = nn.ReLU()
        self._7MaxPooling1D=nn.MaxPool1d(stride=pool_stride_size[1], kernel_size=pool_size[1],padding=1)
        self._8Dropout = nn.Dropout(p=0.5)

        self._401Flattern = nn.Flatten()
        self._41Dense = nn.Linear(in_features=1, out_features=feature_width)
        self._42BatchNormalization = nn.BatchNorm1d(feature_width)
        self._43Tanh = nn.Tanh()
        self._44Dropout = nn.Dropout(p=0.5)
        self.to(self.device)

    def forward(self, x):
        x = self._1Conv1D(x)
        x = self._2BatchNormalization(x)
        x = self._3ReLU(x)
        x = self._4Conv1D(x)
        x = self._5BatchNormalization(x)
        x = self._6ReLU(x)
        x = self._7MaxPooling1D(x)
        x = self._8Dropout(x)
        x = self._401Flattern(x)

        if not hasattr(self, '_initialized'):
            in_features = x.shape[1]
            self._41Dense = nn.Linear(in_features=in_features, out_features=self._41Dense.out_features).to(self.device)
            self._initialized = True

        x = self._41Dense(x)
        x = self._42BatchNormalization(x)
        x = self._43Tanh(x)
        x = self._44Dropout(x)

        return  x

class Classifier(nn.Module):
    def __init__(self,nb_classes=55,nb_layers=2,latent_feature_length=100,use_gpu=False,device="cpu",layer_type='GCN'):
        '''
        :nb_classes:
        :nb_layers:
        :latent_feature_length:
        :layer_type:
        '''
        super(Classifier,self).__init__()
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        self.layer_type = layer_type
        self.latent_feature_length = latent_feature_length
        self.device = device

        self.pkt_length_fextractor = Bert()
        self.arv_time_fextractor = Bert()
        self.stati_charac_fextractor = Feature_extractor(self.latent_feature_length)

        if self.device != "cpu":
          self.arv_time_fextractor = self.arv_time_fextractor.cuda(device)
          self.pkt_length_fextractor = self.pkt_length_fextractor.cuda(device)
          self.stati_charac_fextractor = self.stati_charac_fextractor.cuda(device)

        self._1GATConv = GATConv(in_feats=self.latent_feature_length, out_feats=self.latent_feature_length ,allow_zero_in_degree=True,num_heads=2)
        self._1GATConv.to(torch.device(device))

        self._2GATConv = GATConv(in_feats=self.latent_feature_length * 2, out_feats=self.latent_feature_length,allow_zero_in_degree=True,num_heads=1)
        self._2GATConv.to(torch.device(device))

        self.fc1 = nn.Linear(in_features=self.latent_feature_length * 3, out_features=self.latent_feature_length * 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.classify = nn.Linear(in_features=self.latent_feature_length * 1, out_features=nb_classes)


    def forward(self, g):
        pkt_length_matrix  = self.pkt_length_fextractor(g.ndata['packet_length'].float())
        arv_time_matrix = self.arv_time_fextractor(g.ndata['arrive_time_delta'].float())
        stati_charac_matrix = self.stati_charac_fextractor(g.ndata['statistical_characteristics'].float())

        pkt_length_matrix = self._1GATConv(g,pkt_length_matrix.to(torch.device(self.device)))
        pkt_length_matrix = pkt_length_matrix.view(pkt_length_matrix.size(0), -1) 
        pkt_length_matrix = self._2GATConv(g,pkt_length_matrix.to(torch.device(self.device)))
        pkt_length_matrix = torch.flatten(pkt_length_matrix,1)

        arv_time_matrix = self._1GATConv(g,arv_time_matrix.to(torch.device(self.device)))
        arv_time_matrix = arv_time_matrix.view(arv_time_matrix.size(0), -1)
        arv_time_matrix = self._2GATConv(g,arv_time_matrix.to(torch.device(self.device)))
        arv_time_matrix = torch.flatten(arv_time_matrix,1)

        stati_charac_matrix = self._1GATConv(g,stati_charac_matrix.to(torch.device(self.device)))
        stati_charac_matrix = stati_charac_matrix.view(stati_charac_matrix.size(0), -1)
        stati_charac_matrix = self._2GATConv(g,stati_charac_matrix.to(torch.device(self.device)))
        stati_charac_matrix = torch.flatten(stati_charac_matrix,1)

        g.ndata['packet_length'] = pkt_length_matrix
        g.ndata['arrive_time_delta'] = arv_time_matrix
        g.ndata['statistical_characteristics'] = stati_charac_matrix

        pkt_length_matrix = dgl.mean_nodes(g,'packet_length')
        arv_time_matrix = dgl.mean_nodes(g,'arrive_time_delta')
        stati_charac_matrix = dgl.mean_nodes(g,'statistical_characteristics')

        matrix = torch.cat((pkt_length_matrix,arv_time_matrix,stati_charac_matrix),1)
        matrix = self.fc1(matrix)
        matrix = self.relu1(matrix)
        matrix = self.dropout1(matrix)
        matrix = self.classify(matrix)

        return  matrix

def main(trainset, max_epoch=60, patience=10):
    data_loader = Dataset_load(dumpFilename=trainset)
    model = Classifier(nb_classes=len(data_loader.labelNameSet), latent_feature_length=256, use_gpu=True, device='cuda', layer_type='GAT')
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    model.train()
    epoch_losses = []
    epoch_accuracy_es = []
    batch_size = 32

    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm.trange(max_epoch):
        epoch_loss = 0
        iter = 0
        while data_loader.epoch_over == epoch:
            graphs, labels = data_loader.next_train_batch(batch_size)
            graphs = graphs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            predict_label = model(graphs)
            loss = loss_func(predict_label, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter += 1

        valid_loss = 0
        correct_predictions = 0
        total_predictions = 0
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for _ in range(len(data_loader.valid_index) // batch_size):
                graphs, labels = data_loader.next_valid_batch(batch_size)
                graphs = graphs.to(torch.device('cuda'))
                labels = labels.to(torch.device('cuda'))
                predict_label = model(graphs)
                preds = torch.argmax(predict_label, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss = loss_func(predict_label, labels)
                valid_loss += loss.item()

                preds = torch.argmax(predict_label, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

        valid_loss /= len(data_loader.valid_index) // batch_size
        valid_accuracy = correct_predictions / total_predictions

        print(f'Epoch {epoch+1}/{max_epoch}, Training Loss: {epoch_loss/iter:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
        report = classification_report(all_labels, all_preds, target_names=data_loader.labelNameSet, output_dict=True)
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']

        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"\nMacro Average PR: {macro_avg['precision']:.4f},RC: {macro_avg['recall']:.4f},F1: {macro_avg['f1-score']:.4f}")
        print(f"\nWeighted Average PR: {weighted_avg['precision']:.4f},RC: {weighted_avg['recall']:.4f},F1: {weighted_avg['f1-score']:.4f}")


        # Early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        model.train()

    model.load_state_dict(best_model)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for _ in range(len(data_loader.test_index) // batch_size):
            graphs, labels = data_loader.next_test_batch(batch_size)
            graphs = graphs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            predict_label = model(graphs)
            preds = torch.argmax(predict_label, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nTest Accuracy: {accuracy:.4f}')
    report = classification_report(all_labels, all_preds, target_names=data_loader.labelNameSet, output_dict=True)

    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nMacro Average PR: {macro_avg['precision']:.4f},RC: {macro_avg['recall']:.4f},F1: {macro_avg['f1-score']:.4f}")
    print(f"\nWeighted Average PR: {weighted_avg['precision']:.4f},RC: {weighted_avg['recall']:.4f},F1: {weighted_avg['f1-score']:.4f}")

if __name__ == '__main__':
    main(trainset=r"../dataset/MTA.bin", max_epoch=30)
