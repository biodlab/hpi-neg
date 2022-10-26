#!/usr/bin/env python

import os, sys, argparse, random, pysam, torch, pickle, string
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from itertools import combinations
from Bio.SeqIO.FastaIO import SimpleFastaParser

class dataloader(Dataset):
    def __init__(self, classes, fastafile, enctype = 0, length=2000, embedfile = "vec5_CTC.txt"):
        self.fastafile = fastafile
        self.embedfile = embedfile
        self.length = length
        self.classes = classes
        self.status = []
        self.enctype = enctype

        self.seqs = dict()
        ff = pysam.Fastafile(self.fastafile)
        for prot in ff.references:
            seq = ff.fetch(prot)
            if len(seq) > self.length:
                #print("truncating:", prot) 
                seq = seq[:self.length]
            self.seqs[prot] = seq
        ff.close()

        self.status = np.array([x[2] for x in self.classes])

        self.negsamples = len(np.where(self.status == 0)[0])
        self.possamples = len(np.where(self.status == 1)[0])
        print("Loaded:", self.possamples, "positives", self.negsamples, "negatives")
        if self.embedfile is not None and self.enctype == 0:
            self.load_embeddings()

    def sampler(self):
        class_count = np.array([self.negsamples, self.possamples])
        print("class_count:", class_count)
        class_count = 1./class_count
        sample_weights = torch.from_numpy(np.array([class_count[x] for x in self.status]))
        return sample_weights

    def load_embeddings(self):
        self.embdim = None
        self.amembeddings = dict()
        for line in open(self.embedfile):
            s = line.strip().split("\t")
            v = np.array([float(x) for x in s[1].split()])
            if self.embdim is None: self.embdim = len(v)
            self.amembeddings[s[0]] = v

        self.embeddings = np.zeros((len(self.seqs), self.length, self.embdim)).astype(np.float32)
        for i, seq in enumerate(self.seqs):
            self.embeddings[i,:len(self.seqs[seq].strip())]  = np.array([self.amembeddings[x] for x in self.seqs[seq].strip()])
            self.seqs[seq] = i

    def __getitem__(self, index):
        status = self.classes[index][2]
        if self.embedfile is not None and self.enctype == 0:
            seq1 = self.embeddings[self.seqs[self.classes[index][0]]]
            seq2 = self.embeddings[self.seqs[self.classes[index][1]]]
        elif self.enctype == 3:
            alphabet =['B','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','U','X']
            seq1 = [alphabet.index(x)+1 for x in self.seqs[self.classes[index][0]]]
            seq2 = [alphabet.index(x)+1 for x in self.seqs[self.classes[index][0]]]
            for i in range(self.length - len(seq1)):
                seq1.append(0)
            for i in range(self.length - len(seq2)):
                seq2.append(0)
            seq1 = torch.nn.functional.one_hot(torch.tensor(seq1), 24).float()
            seq2 = torch.nn.functional.one_hot(torch.tensor(seq2), 24).float()
        elif self.enctype == 1:
            alphabet = [x for x in string.ascii_uppercase]
            seq1 = [alphabet.index(x)+1 for x in self.seqs[self.classes[index][0]]]
            seq2 = [alphabet.index(x)+1 for x in self.seqs[self.classes[index][0]]]
            for i in range(self.length - len(seq1)):
                seq1.append(0)
            for i in range(self.length - len(seq2)):
                seq2.append(0)
            seq1 = torch.nn.functional.one_hot(torch.tensor(seq1), 27).float()
            seq2 = torch.nn.functional.one_hot(torch.tensor(seq2), 27).float()

        else:
            print("Unknown encoding type:", self.enctype)
            sys.exit(1)
        return (seq1, seq2, status)

    def __len__(self):
        return len(self.classes)

# TODO: for feature_file need both Feat.npy for hpidb and DenovoFeat.npy
# Feat.npy is hpidb and DenovoFeat.npy is orig denovo set
def get_denovo_svm_data(classes, feature_file="denovo-hpidb-features.npy"):
    seq_feat = []
    labels = []
    t1 = np.load(feature_file,allow_pickle=True)
    host_name = np.array(classes[:,0])
    pathogen_name = np.array(classes[:,1])
    interact = np.array(classes[:,2].astype(np.int))
    list1=[]
    for i in range(len(host_name)):
        key1 = str(host_name[i])
        key2 = str(pathogen_name[i])
        first = t1.item().get(key1)[0]
        sec = t1.item().get(key2)[0]
        final = np.concatenate((first,sec))
        seq_feat.append(final)
        labels.append(interact[i])
    seq_feat=np.array(seq_feat)
    labels=np.array(labels)
    return seq_feat,labels

def split_data(classes, trainsize = 0.8):
    random.shuffle(classes)
    z = np.array(classes)
    z2 = np.array(z[:,2].astype(np.int))
    whereneg = np.where(z2 == 0)
    wherepos = np.where(z2 == 1)
    trainnegind = whereneg[0][:int(len(whereneg[0])*trainsize)]
    trainposind = wherepos[0][:int(len(wherepos[0])*trainsize)]
    validnegind = whereneg[0][int(len(whereneg[0])*trainsize):]
    validposind = wherepos[0][int(len(wherepos[0])*trainsize):]
    if len(trainnegind) + len(trainposind) + len(validnegind) + len(validposind) != len(classes):
        print("ERROR: split data mismatch")
        sys.exit(1)
    trainclasses = z[(np.concatenate((trainnegind,trainposind)))]
    validclasses = z[(np.concatenate((validnegind,validposind)))]
    if len(trainclasses) + len(validclasses) != len(classes):
        print("ERROR: split data mismatch")
        sys.exit(1)
    tc = []
    vc = []
    for item in trainclasses:
        tc.append([item[0], item[1], int(item[2])])
    for item in validclasses:
        vc.append([item[0], item[1], int(item[2])])
    return tc, vc

def kfold_split(classes, partitions = 5):
    random.shuffle(classes)
    z = np.array(classes)
    z2 = np.array(z[:,2].astype(np.int))
    whereneg = np.where(z2 == 0)
    wherepos = np.where(z2 == 1)
    print(len(whereneg), len(whereneg[0]), type(whereneg[0]), whereneg[0][:10])
    print(len(wherepos), len(wherepos[0]), type(wherepos[0]), wherepos[0][:10])
    psize = np.int(len(wherepos[0]) / partitions)
    nsize = np.int(len(whereneg[0]) / partitions)
    for i in range(partitions):
        trainnegind = np.array([]).astype(np.int)
        trainposind = np.array([]).astype(np.int)
        validnegind = np.array([]).astype(np.int)
        validposind = np.array([]).astype(np.int)
        for j in range(partitions):
            if j != i:
                trainnegind = np.concatenate((trainnegind, whereneg[0][j*nsize:j*nsize+nsize]))
                trainposind = np.concatenate((trainposind, wherepos[0][j*psize:j*psize+psize]))
            else:
                validnegind = whereneg[0][j*nsize:j*nsize+nsize]
                validposind = wherepos[0][j*psize:j*psize+psize]
        trainclasses = z[(np.concatenate((trainnegind,trainposind)))]
        validclasses = z[(np.concatenate((validnegind,validposind)))]
        tc = [[item[0], item[1], int(item[2])] for item in trainclasses]
        vc = [[item[0], item[1], int(item[2])] for item in validclasses]
        random.shuffle(tc)
        random.shuffle(vc)
        yield tc, vc

class deepviral_network(nn.Module):
    def __init__(self, hidden_dim=150,one_hot=True):
        super().__init__()
        self.kernel_rate_1=0.16
        self.kernel_rate_2=0.14
        self.strides_rate_1=0.15
        self.strides_rate_2=0.25
        self.one_hot=one_hot
        self.encoding_layer=nn.Embedding(24,15)
        in_channels=24
        out_channels=16 
        self.conv1=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=8)
        self.conv2=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=16)
        self.conv3=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=24)
        self.conv4=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=32)
        self.conv5=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=40)
        self.conv6=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=48)
        self.conv7=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=56)
        self.conv8=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=64)
        self.drop1=nn.Dropout(0.5)
        self.drop2=nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool1d(200)
        #self.fc1 = nn.Linear(1152, 8)
        self.fc1 = nn.Linear(512, 8)
        self.fc2 = nn.Linear(1, 1)
        self.sm = torch.nn.Softmax(dim=1)
        self._initialize_weights()
    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        torch.nn.init.xavier_uniform_(self.conv7.weight)
        torch.nn.init.xavier_uniform_(self.conv8.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.zeros_(self.conv4.bias)
        torch.nn.init.zeros_(self.conv5.bias)
        torch.nn.init.zeros_(self.conv6.bias)
        torch.nn.init.zeros_(self.conv7.bias)
        torch.nn.init.zeros_(self.conv8.bias)
    def helper_conv(self,x,conv1):
        
        x1=conv1(x.transpose(2,1))
        x1=self.max_pool1(x1)
        x1=torch.flatten(x1,1)
        return x1
    def seq(self, x, debug=False):
        
        x1=self.helper_conv(x,self.conv1)
        x2=self.helper_conv(x,self.conv2)
        x3=self.helper_conv(x,self.conv3)
        x4=self.helper_conv(x,self.conv4)
        x5=self.helper_conv(x,self.conv5)
        x6=self.helper_conv(x,self.conv6)
        x7=self.helper_conv(x,self.conv7)
        x8=self.helper_conv(x,self.conv8)
        X1=torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        X1=self.fc1(X1)
        
        X1=torch.nn.functional.leaky_relu(X1,0.1)
        X1=self.drop2(X1)
        return X1
            
    def forward(self, x1, x2, softmax=False, debug=False):
        if self.one_hot==False:
            x1=self.encoding_layer(x1)
            x2=self.encoding_layer(x2)
            x1=F.dropout2d(x1.permute(0,2,1),self.embed_drop,training=self.training) 
            x2=F.dropout2d(x2.permute(0,2,1),self.embed_drop,training=self.training) 
        X1= self.seq(x1)
        X2 = self.seq(x2)
        row,col=X1.shape
        final=torch.bmm(X1.view(row,1,col),X2.view(row,col,1))
        final=torch.squeeze(final,2)
        final=self.fc2(final) 
        final=torch.sigmoid(final)
        final=torch.squeeze(final)
        return final

class deeptrio_network(nn.Module):
    def __init__(self, hidden_dim=150):
        super().__init__()
        self.kernel_rate_1=0.16
        self.kernel_rate_2=0.14
        self.strides_rate_1=0.15
        self.strides_rate_2=0.25
        self.conv1=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*2**2)),stride=int(np.ceil(self.strides_rate_1*(2-1))),bias=False)
        self.conv2=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*3**2)),stride=int(np.ceil(self.strides_rate_1*(3-1))),bias=False)
        self.conv3=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*4**2)),stride=int(np.ceil(self.strides_rate_1*(4-1))),bias=False)
        self.conv4=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*5**2)),stride=int(np.ceil(self.strides_rate_1*(5-1))),bias=False)
        self.conv5=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*6**2)),stride=int(np.ceil(self.strides_rate_1*(6-1))),bias=False)
        self.conv6=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*7**2)),stride=int(np.ceil(self.strides_rate_1*(7-1))),bias=False)
        self.conv7=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*8**2)),stride=int(np.ceil(self.strides_rate_1*(8-1))),bias=False)
        self.conv8=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*9**2)),stride=int(np.ceil(self.strides_rate_1*(9-1))),bias=False)
        self.conv9=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*10**2)),stride=int(np.ceil(self.strides_rate_1*(10-1))),bias=False)
        self.conv10=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*11**2)),stride=int(np.ceil(self.strides_rate_1*(11-1))),bias=False)
        self.conv11=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*12**2)),stride=int(np.ceil(self.strides_rate_1*(12-1))),bias=False)
        self.conv12=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*13**2)),stride=int(np.ceil(self.strides_rate_1*(13-1))),bias=False)
        self.conv13=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*14**2)),stride=int(np.ceil(self.strides_rate_1*(14-1))),bias=False)
        self.conv14=nn.Conv1d(in_channels=27,out_channels=150,kernel_size=int(np.ceil(self.kernel_rate_1*15**2)),stride=int(np.ceil(self.strides_rate_1*(15-1))),bias=False)
        self.conv15=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*16**2)),stride=int(np.ceil(self.strides_rate_2*(16-1))),bias=False)
        self.conv16=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*17**2)),stride=int(np.ceil(self.strides_rate_2*(17-1))),bias=False)
        self.conv17=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*18**2)),stride=int(np.ceil(self.strides_rate_2*(18-1))),bias=False)
        self.conv18=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*19**2)),stride=int(np.ceil(self.strides_rate_2*(19-1))),bias=False)
        self.conv19=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*20**2)),stride=int(np.ceil(self.strides_rate_2*(20-1))),bias=False)
        self.conv20=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*21**2)),stride=int(np.ceil(self.strides_rate_2*(21-1))),bias=False)
        self.conv21=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*22**2)),stride=int(np.ceil(self.strides_rate_2*(22-1))),bias=False)
        self.conv22=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*23**2)),stride=int(np.ceil(self.strides_rate_2*(23-1))),bias=False)
        self.conv23=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*24**2)),stride=int(np.ceil(self.strides_rate_2*(24-1))),bias=False)
        self.conv24=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*25**2)),stride=int(np.ceil(self.strides_rate_2*(25-1))),bias=False)
        self.conv25=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*26**2)),stride=int(np.ceil(self.strides_rate_2*(26-1))),bias=False)
        self.conv26=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*27**2)),stride=int(np.ceil(self.strides_rate_2*(27-1))),bias=False)
        self.conv27=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*28**2)),stride=int(np.ceil(self.strides_rate_2*(28-1))),bias=False)
        self.conv28=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*29**2)),stride=int(np.ceil(self.strides_rate_2*(29-1))),bias=False)
        self.conv29=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*30**2)),stride=int(np.ceil(self.strides_rate_2*(30-1))),bias=False)
        self.conv30=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*31**2)),stride=int(np.ceil(self.strides_rate_2*(31-1))),bias=False)
        self.conv31=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*32**2)),stride=int(np.ceil(self.strides_rate_2*(32-1))),bias=False)
        self.conv32=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*33**2)),stride=int(np.ceil(self.strides_rate_2*(33-1))),bias=False)
        self.conv33=nn.Conv1d(in_channels=27,out_channels=175,kernel_size=int(np.ceil(self.kernel_rate_2*34**2)),stride=int(np.ceil(self.strides_rate_2*(34-1))),bias=False)
        self.drop1=nn.Dropout(0.2)
        self.drop2=nn.Dropout(0.1)
        self.fc1 = nn.Linear(5425, 256)
        self.fc2 = nn.Linear(256, 2)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        torch.nn.init.xavier_uniform_(self.conv7.weight)
        torch.nn.init.xavier_uniform_(self.conv8.weight)
        torch.nn.init.xavier_uniform_(self.conv9.weight)
        torch.nn.init.xavier_uniform_(self.conv10.weight)
        torch.nn.init.xavier_uniform_(self.conv11.weight)
        torch.nn.init.xavier_uniform_(self.conv12.weight)
        torch.nn.init.xavier_uniform_(self.conv13.weight)
        torch.nn.init.xavier_uniform_(self.conv14.weight)
        torch.nn.init.xavier_uniform_(self.conv15.weight)
        torch.nn.init.xavier_uniform_(self.conv16.weight)
        torch.nn.init.xavier_uniform_(self.conv17.weight)
        torch.nn.init.xavier_uniform_(self.conv18.weight)
        torch.nn.init.xavier_uniform_(self.conv19.weight)
        torch.nn.init.xavier_uniform_(self.conv20.weight)
        torch.nn.init.xavier_uniform_(self.conv21.weight)
        torch.nn.init.xavier_uniform_(self.conv22.weight)
        torch.nn.init.xavier_uniform_(self.conv23.weight)
        torch.nn.init.xavier_uniform_(self.conv24.weight)
        torch.nn.init.xavier_uniform_(self.conv25.weight)
        torch.nn.init.xavier_uniform_(self.conv26.weight)
        torch.nn.init.xavier_uniform_(self.conv27.weight)
        torch.nn.init.xavier_uniform_(self.conv28.weight)
        torch.nn.init.xavier_uniform_(self.conv29.weight)
        torch.nn.init.xavier_uniform_(self.conv30.weight)
        torch.nn.init.xavier_uniform_(self.conv31.weight)
        torch.nn.init.xavier_uniform_(self.conv32.weight)
        torch.nn.init.xavier_uniform_(self.conv33.weight)

    def helper_conv(self,x,conv1):
        x1=conv1(x.transpose(2,1))
        x1=torch.nn.functional.dropout2d(x1, 0.05, training=self.training)
        x1=torch.nn.functional.relu(x1)
        x1 = torch.nn.functional.max_pool1d(x1, x1.shape[2])
        x1=torch.squeeze(x1,axis=2)
        return x1

    def seq(self, x, debug=False):
        x1 = torch.cat((self.helper_conv(x,self.conv1), self.helper_conv(x,self.conv2), self.helper_conv(x,self.conv3), self.helper_conv(x,self.conv4), self.helper_conv(x,self.conv5), self.helper_conv(x,self.conv6), self.helper_conv(x,self.conv7), self.helper_conv(x,self.conv8), self.helper_conv(x,self.conv9), self.helper_conv(x,self.conv10), self.helper_conv(x,self.conv11), self.helper_conv(x,self.conv12), self.helper_conv(x,self.conv13), self.helper_conv(x,self.conv14)), dim=1)
        
        x2=torch.cat((self.helper_conv(x,self.conv15), self.helper_conv(x,self.conv16), self.helper_conv(x,self.conv17), self.helper_conv(x,self.conv18), self.helper_conv(x,self.conv19), self.helper_conv(x,self.conv20), self.helper_conv(x,self.conv21), self.helper_conv(x,self.conv22), self.helper_conv(x,self.conv23)), dim=1)
        
        x3=torch.cat((self.helper_conv(x,self.conv24), self.helper_conv(x,self.conv25), self.helper_conv(x,self.conv26), self.helper_conv(x,self.conv27), self.helper_conv(x,self.conv28), self.helper_conv(x,self.conv29), self.helper_conv(x,self.conv30), self.helper_conv(x,self.conv31), self.helper_conv(x,self.conv32), self.helper_conv(x,self.conv33)), dim=1)
        return torch.cat((x1, x2, x3), dim=1)
            
    def forward(self, x1, x2):
        x11 = self.seq(x1)
        x21 = self.seq(x2)
        res=x11+x21
        res=self.drop1(res)
        res = self.fc1(res)
        res=self.drop2(res)

        res=torch.nn.functional.relu(res)
        res=self.fc2(res)
        return res

class conv_network(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.conv1 = nn.Conv1d(27, hidden_dim, 7)
        self.m1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(128,256 ,4)
        self.m2 = nn.MaxPool1d(3)
        self.gap = nn.AvgPool1d(20)

        self.fc1 = nn.Linear(256*2, 2)
        self.sm = torch.nn.Softmax(dim=1)
        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)


    def seq(self, x):
        x = self.conv1(x.transpose(2,1))
        x = self.conv2(x)#.transpose(2, 1))
        x=self.gap(x)
        x = torch.squeeze(x)
        return x

    def forward(self, x1, x2):
        x1 = self.seq(x1)
        x2 = self.seq(x2)
        x =torch.cat([x1.transpose(1,2), x2.transpose(1,2)],dim=2)
        
        x = torch.mean(x, 1, True)
        x = torch.squeeze(x, 1)
        x = self.fc1(x)
        return x
 


class pipr_network(nn.Module):
    def __init__(self, hidden_dim=50):
        super().__init__()
        
        self.conv1 = nn.Conv1d(13, hidden_dim, 3)
        self.m1 = nn.MaxPool1d(3)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv2 = nn.Conv1d(150, hidden_dim, 3)
        self.m2 = nn.MaxPool1d(3)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv3 = nn.Conv1d(150, hidden_dim, 3)
        self.m3 = nn.MaxPool1d(3)
        self.gru3 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv4 = nn.Conv1d(150, hidden_dim, 3)
        self.m4 = nn.MaxPool1d(3)
        self.gru4 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv5 = nn.Conv1d(150, hidden_dim, 3)
        self.m5 = nn.MaxPool1d(3)
        self.gru5 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv6 = nn.Conv1d(150, hidden_dim, 3)
        self.gap = nn.AvgPool1d(5)

        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, int((hidden_dim+7)/2))
        self.fc3 = nn.Linear(int((hidden_dim+7)/2), 2)
        self.sm = torch.nn.Softmax(dim=1)
        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.zeros_(self.conv4.bias)
        torch.nn.init.zeros_(self.conv5.bias)
        torch.nn.init.zeros_(self.conv6.bias)
        for gru in [self.gru1, self.gru2, self.gru3, self.gru4, self.gru5]:
            for name, param in gru.named_parameters():
                if "weight" in name:
                    torch.nn.init.orthogonal_(param)
                if "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)
                if "weight_ih" in name:
                    torch.nn.init.orthogonal_(param)
                if "bias_hh" in name:
                    torch.nn.init.zeros_(param)
                if "bias_ih" in name:
                    torch.nn.init.zeros_(param)

    def seq(self, x):
        x = self.conv1(x.transpose(2,1))
        x = self.m1(x)
        x = x.transpose(2,1)
        x = torch.cat((self.gru1(x)[0], x), dim=2)
        x = self.conv2(x.transpose(2, 1))
        x = self.m2(x)
        x = x.transpose(2,1)
        x = torch.cat((self.gru2(x)[0], x), dim=2)
        x = self.conv3(x.transpose(2,1))
        x = self.m3(x)
        x = x.transpose(2,1)
        x = torch.cat((self.gru3(x)[0], x), dim=2)
        x = self.conv4(x.transpose(2,1))
        x = self.m4(x)
        x = x.transpose(2,1)
        x = torch.cat((self.gru4(x)[0], x), dim=2)
        x = self.conv5(x.transpose(2,1))
        x = self.m5(x)
        x = x.transpose(2,1)
        x = torch.cat((self.gru5(x)[0], x), dim=2)
        x = self.conv6(x.transpose(2,1))
        x = self.gap(x)
        x = torch.squeeze(x)
        return x

    def forward(self, x1, x2):
        x1 = self.seq(x1)
        x2 = self.seq(x2)
        x = x1 * x2
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x, .3)
        x = self.fc2(x)
        x = torch.nn.functional.leaky_relu(x, .3)
        x = self.fc3(x)
        return x

def train_denovo(args, datadict, trainclasses, validclasses, foldnumber=-1):
    featfile = "denovo-features.npy"
    if args.fastafile.startswith("hpidb"):
        featfile = "denovo-hpidb-features.npy"
    trainfeat,trainlabels=get_denovo_svm_data(trainclasses, feature_file=featfile)
    validfeat,validlabels=get_denovo_svm_data(validclasses, feature_file=featfile)
    clf=SVC(degree=3,gamma='auto',C=1,coef0=0,cache_size=100, verbose=True, probability=True)
    res = clf.fit(trainfeat, trainlabels)
    resultdata = clf.predict_proba(validfeat)
    y_score = np.argmax(resultdata, axis=1)
    validacc = sklearn.metrics.accuracy_score(validlabels, y_score)
    validauc = sklearn.metrics.roc_auc_score(validlabels, resultdata[:,1])
    precision,recall,thres=sklearn.metrics.precision_recall_curve(validlabels, resultdata[:,1])
    validaupr=sklearn.metrics.auc(recall,precision)
    validbal = sklearn.metrics.balanced_accuracy_score(validlabels, y_score)
    validmat = sklearn.metrics.matthews_corrcoef(validlabels, y_score)
    resulttruth = validlabels
    #print(validacc, validauc, "res:", resultdata)

    foldname = ""
    testtype = "valid"
    if foldnumber >= 0: foldname = "f"+str(foldnumber)+"-"
    if foldnumber == -2: testtype = "test"
    datadict[foldname+testtype+"_resultdata"] = resultdata
    datadict[foldname+testtype+"_resulttruth"] = resulttruth
    datadict[foldname+testtype+"auc"] = validauc
    datadict[foldname+testtype+"acc"] = validacc
    datadict[foldname+testtype+"aupr"] = validaupr
    datadict[foldname+testtype+"bal"] = validbal
    datadict[foldname+testtype+"mat"] = validmat
    save_data(args, datadict)

def generate_splits(args, dotrain=True):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainloss = []
    validloss = []
    trainacc = []
    validacc = []
    trainauc = []
    validauc = []
    trainaupr = []
    validaupr = []
    trainbal = []
    validbal = []
    trainmat = []
    validmat = []
    testloss = []
    testacc = []
    testauc = []
    datadict = dict()

    if args.fivefold:
        classes = [[x.strip().split(",")[0], x.strip().split(",")[1], int(x.strip().split(",")[2])] for x in open(args.trainfile,"r").readlines()]
        random.shuffle(classes)
        print("Running five-fold cross-validation, loaded:", len(classes), "entries")

        from sklearn.model_selection import KFold, ShuffleSplit
        kf = KFold(n_splits=5, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        foldnumber = 0
        for trainclasses, validclasses in  kfold_split(classes):
            print("split - train:", len(trainclasses), "valid:", len(validclasses))

            if args.model == "pipr":
                model = pipr_network().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
                enctype = 0
            elif args.model == "conv":
                model = conv_network().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                enctype = 1
            elif args.model == "denovo":
                print("denovo: running fold", foldnumber)
                enctype = 2
                train_denovo(args, datadict, np.array(trainclasses), np.array(validclasses), foldnumber=foldnumber)
                foldnumber+=1
                continue
            elif args.model == "deepviral":
                print("Running deepviral")
                model = deepviral_network().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                enctype = 3
                criterion = torch.nn.BCELoss()
            elif args.model == "deeptrio":
                print("Running deeptrio")
                enctype = 1
                model = deeptrio_network().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            print(model)
            data = dataloader(trainclasses, fastafile=args.fastafile, length=args.maxlength, enctype=enctype)
            data_loader = DataLoader(dataset=data, batch_size=args.batchsize, shuffle=True)
            valid_data = dataloader(validclasses, fastafile=args.fastafile, length=args.maxlength, enctype=enctype)
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batchsize, shuffle=False)
            trloss, vlloss, tracc, vlacc, trauc, vlauc, traupr, vlaupr, trbal, vlbal, trmat, vlmat, _, _ , _, _, _, _ = train(args, datadict, device, model, optimizer, criterion, data_loader, valid_loader, None, foldnumber = foldnumber)
            trainloss.append(trloss)
            validloss.append(vlloss)
            trainacc.append(tracc)
            validacc.append(vlacc)
            trainauc.append(trauc)
            validauc.append(vlauc)
            trainaupr.append(traupr)
            validaupr.append(vlaupr)
            trainbal.append(trbal)
            validbal.append(vlbal)
            trainmat.append(trmat)
            validmat.append(vlmat)
            datadict["trainloss"] = trloss
            datadict["validloss"] = validloss
            datadict["trainacc"] = trainacc
            datadict["validacc"] = validacc
            datadict["trainauc"] = trainauc
            datadict["validauc"] = validauc
            datadict["trainaupr"] = trainaupr
            datadict["validaupr"] = validaupr
            datadict["trainbal"] = trainbal
            datadict["validbal"] = validbal
            datadict["trainmat"] = trainmat
            datadict["validmat"] = validmat
            save_data(args, datadict)
            print("Deleting model")
            del model
            del optimizer
            foldnumber+=1
    else:
        trainclasses = [[x.strip().split(",")[0], x.strip().split(",")[1], int(x.strip().split(",")[2])] for x in open(args.trainfile,"r").readlines()]
        random.shuffle(trainclasses)
        testclasses = [[x.strip().split(",")[0], x.strip().split(",")[1], int(x.strip().split(",")[2])] for x in open(args.testfile,"r").readlines()]
        random.shuffle(testclasses)
        print("Print running train/test, loaded:", len(trainclasses), "train entries", len(testclasses), "test entries")

        criterion = torch.nn.CrossEntropyLoss()
        if args.model == "pipr":
            model = pipr_network().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
            enctype = 0
        elif args.model == "conv":
            model = conv_network().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            enctype = 1
        elif args.model == "denovo":
            print("Running denovo")
            enctype = 2
            train_denovo(args, datadict, np.array(trainclasses), np.array(testclasses), foldnumber = -2)
        elif args.model == "deepviral":
            print("Running deepviral")
            model = deepviral_network().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            enctype = 3
            criterion = torch.nn.BCELoss()
        elif args.model == "deeptrio":
            print("Running deeptrio")
            enctype = 1
            model = deeptrio_network().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


        if (enctype != 2):
            print(model)

            trainclasses, validclasses = split_data(trainclasses)

            data = dataloader(trainclasses, fastafile=args.fastafile, length=args.maxlength, enctype=enctype)
            data_loader = DataLoader(dataset=data, batch_size=args.batchsize, shuffle=True)
            valid_data = dataloader(validclasses, fastafile=args.fastafile, length=args.maxlength, enctype=enctype)
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batchsize, shuffle=False)
            test_data = dataloader(testclasses, fastafile=args.fastafile, length=args.maxlength, enctype=enctype)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
            trainloss, validloss, trainacc, validacc, trainauc, validauc, trainaupr, validaupr, trainbal, validbal, trainmat, validmat, tloss, tacc, tauc, taupr, tbal, tmat = train(args, datadict, device, model, optimizer, criterion, data_loader, valid_loader, test_loader)
            datadict["trainloss"] = trainloss
            datadict["validloss"] = validloss
            datadict["trainacc"] = trainacc
            datadict["validacc"] = validacc
            datadict["trainauc"] = trainauc
            datadict["validauc"] = validauc
            datadict["trainaupr"] = trainaupr
            datadict["validaupr"] = validaupr
            datadict["trainbal"] = trainbal
            datadict["validbal"] = validbal
            datadict["trainmat"] = trainmat
            datadict["validmat"] = validmat
            datadict["testloss"] = tloss
            datadict["testacc"] = tacc
            datadict["testauc"] = tauc
            datadict["testaupr"] = taupr
            datadict["testbal"] = tbal
            datadict["testmat"] = tmat
            save_data(args, datadict)

def save_data(args, datadict):
    outfile=args.name+".pickle"
    f = open(outfile, "wb")
    pickle.dump(datadict, f)
    f.close()

def train(args, datadict, device, model, optimizer, criterion, dataloader, validloader, testloader, foldnumber = -1):
    trainloss = []
    validloss = []
    trainacc = []
    validacc = []
    trainauc = []
    validauc = []
    trainaupr = []
    validaupr = []
    trainbal = []
    validbal = []
    trainmat = []
    validmat = []

    for epoch in range(args.epochs):
        model.train()
        loopcount = 0
        totalloss = 0

        y_true = np.array([])
        y_score = np.array([])

        resultdata = np.zeros((0,2)).astype(np.float32)
        resulttruth = np.zeros((0)).astype(np.int64)
        for i, (seq1, seq2, status) in enumerate(dataloader):
            seq1 = seq1.to(device, non_blocking=True)
            seq2 = seq2.to(device, non_blocking=True)
            resulttruth = np.concatenate((resulttruth, status))
            status = status.to(device, non_blocking=True)
    
            optimizer.zero_grad()
            ret = model(seq1, seq2)

            if args.model == "deepviral":
                loss = criterion(ret, status.float())
            else:
                loss = criterion(ret, status)

            loss.backward()

            optimizer.step()

            if args.model == "deepviral":
                maxes=torch.tensor([0 if (1-ret1)>ret1 else 1 for ret1 in ret  ]).to(device,non_blocking=True)
                tmp = torch.tensor([[1-ret1, ret1] for ret1 in ret]).float()
                resultdata = np.concatenate((resultdata, tmp.cpu().detach().numpy()))
            else:
                maxes = torch.argmax(ret, dim=1)
                resultdata = np.concatenate((resultdata, ret.cpu().detach().numpy()))
            count=len(torch.where(status==maxes)[0])
            a0 = len(torch.where(status == 0)[0])
            a1 = len(torch.where(status == 1)[0])

            print("loss:", loss.item(), epoch, i, str(a0)+"/"+str(a1), count)
            totalloss+=loss.cpu().detach().numpy()
            loopcount+=1

            y_true = np.append(y_true, status.cpu().detach().numpy())
            y_score = np.append(y_score, maxes.cpu().detach().numpy())

        probs = torch.nn.functional.softmax(torch.tensor(resultdata))
        acc = sklearn.metrics.accuracy_score(y_true, y_score)
        auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        newauc = sklearn.metrics.roc_auc_score(y_true, probs[:,1])
        precision,recall,thres=sklearn.metrics.precision_recall_curve(y_true,probs[:,1])
        aupr=sklearn.metrics.auc(recall,precision)
        balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_score)
        mathews_coef = sklearn.metrics.matthews_corrcoef(y_true,y_score)
         
        trainacc.append(acc)
        trainauc.append(auc)
        trainloss.append(totalloss/loopcount)
        trainaupr.append(aupr)
        trainbal.append(balanced_acc)
        trainmat.append(mathews_coef)
        print("Train loss:", trainloss[-1], "ACC:", trainacc[-1], "AUC:", trainauc[-1], "AUPR:", trainaupr[-1], "bal:", trainbal[-1], "MAT:", trainmat[-1])

        vloss, vauc, vacc, vaupr, vbal, vmat = validate(args, datadict, epoch, model, device, criterion, validloader, foldnumber = foldnumber)
        validloss.append(vloss)
        validauc.append(vauc)
        validacc.append(vacc)
        validaupr.append(vaupr)
        validbal.append(vbal)
        validmat.append(vmat)
        print("Valid loss:", validloss[-1], "ACC:", validacc[-1], "AUC:", validauc[-1], "AUPR:", validaupr[-1], "bal:", validbal[-1], "MAT:", validmat[-1])
        sys.stdout.flush()
        foldname = ""
        if foldnumber != -1: foldname = "f"+str(foldnumber)+"-"
        datadict[foldname+"train_"+str(epoch)+"_resultdata"] = resultdata
        datadict[foldname+"train_"+str(epoch)+"_resulttruth"] = resulttruth
        datadict[foldname+"trainacc"] = trainacc
        datadict[foldname+"trainauc"] = trainauc
        datadict[foldname+"trainloss"] = trainloss
        datadict[foldname+"trainaupr"] = trainaupr
        datadict[foldname+"trainbal"] = trainbal
        datadict[foldname+"trainmat"] = trainmat
        datadict[foldname+"validloss"] = validloss
        datadict[foldname+"validauc"] = validauc
        datadict[foldname+"validacc"] = validacc
        datadict[foldname+"validaupr"] = validaupr
        datadict[foldname+"validbal"] = validbal
        datadict[foldname+"validmat"] = validmat
        save_data(args, datadict)

    print("Train loss:", trainloss)
    print("Train ACC:", trainacc)
    print("Train AUC:", trainauc)
    print("Train AUPR:", trainaupr)
    print("Train bal:", trainbal)
    print("Train MAT:", trainmat)
    print("Valid loss:", validloss)
    print("Valid ACC:", validacc)
    print("Valid AUC:", validauc)
    print("Valid AUPR:", validaupr)
    print("Valid bal:", validbal)
    print("Valid MAT:", validmat)

    tloss = -1
    tacc = -1
    tauc = -1
    taupr = -1
    tbal = -1
    tmat = -1
    if not args.fivefold and testloader is not None:
        print("Running test set")
        tloss, tauc, tacc, taupr, tbal, tmat = validate(args, datadict, 0, model, device, criterion, testloader, foldnumber = -2)
        print("Test loss:", tloss, "AUC:", tauc, "ACC:", tacc, "AUPR:", taupr, "bal:", tbal, "MAT:", tmat)
        datadict["testloss"] = tloss
        datadict["testauc"] = tauc
        datadict["testacc"] = tacc
        datadict["testaupr"] = taupr
        datadict["testbal"] = tbal
        datadict["testmat"] = tmat
    return trainloss, validloss, trainacc, validacc, trainauc, validauc, trainaupr, validaupr, trainbal, validbal, trainmat, validmat, tloss, tacc, tauc, taupr, tbal, tmat


def validate(args, datadict, epoch, model, device, criterion, valid_loader, foldnumber = -1):
    model.eval()
    loopcount = 0
    totalloss = 0

    y_true = np.array([])
    y_score = np.array([])

    resultdata = np.zeros((0,2)).astype(np.float32)
    resulttruth = np.zeros((0)).astype(np.int64)
    for i, (seq1, seq2, status) in enumerate(valid_loader):
        seq1 = seq1.to(device, non_blocking=True)
        seq2 = seq2.to(device, non_blocking=True)
        resulttruth = np.concatenate((resulttruth, status))
        status = status.to(device, non_blocking=True)

        ret = model(seq1, seq2)
        if args.model == "deepviral":
            loss = criterion(ret, status.float())
        else:
            loss = criterion(ret, status)

        if args.model == "deepviral":
            maxes=torch.tensor([0 if (1-ret1)>ret1 else 1 for ret1 in ret  ]).to(device,non_blocking=True)
            tmp = torch.tensor([[1-ret1, ret1] for ret1 in ret]).float()
            resultdata = np.concatenate((resultdata, tmp.cpu().detach().numpy()))
        else:
            maxes = torch.argmax(ret, dim=1)
            resultdata = np.concatenate((resultdata, ret.cpu().detach().numpy()))
        count=len(torch.where(status==maxes)[0])
        a0 = len(torch.where(status == 0)[0])
        a1 = len(torch.where(status == 1)[0])
        
        totalloss+=loss.cpu().detach().numpy()
        loopcount+=1

        y_true = np.append(y_true, status.cpu().detach().numpy())
        y_score = np.append(y_score, maxes.cpu().detach().numpy())

        print("valid loss:", loss.item(), i, str(a0)+"/"+str(a1), count)

    probs = torch.nn.functional.softmax(torch.tensor(resultdata))
    acc = sklearn.metrics.accuracy_score(y_true, y_score)
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)
    newauc = sklearn.metrics.roc_auc_score(y_true, probs[:,1])
    precision,recall,thres=sklearn.metrics.precision_recall_curve(y_true,probs[:,1])
    aupr=sklearn.metrics.auc(recall,precision)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_score)
    mathews_coef = sklearn.metrics.matthews_corrcoef(y_true,y_score)

    foldname = ""
    testtype = "valid"
    if foldnumber >=0: foldname = "f"+str(foldnumber)+"-"
    if foldnumber == -2: testtype = "test"
    datadict[foldname+testtype+"_"+str(epoch)+"_resultdata"] = resultdata
    datadict[foldname+testtype+"_"+str(epoch)+"_resulttruth"] = resulttruth
    return(totalloss/loopcount), auc, acc, aupr, balanced_acc, mathews_coef

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", default=256, type=int, help="default: 256")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="default: 100")
    parser.add_argument("-d", "--device", default="cuda", type=str)
    parser.add_argument("-m", "--model", default="pipr", type=str, help="model (default: pipr)")
    parser.add_argument("-l", "--maxlength", default=2000, type=int, help="default: 2000")
    parser.add_argument("-f", "--fastafile", default=None, type=str)
    parser.add_argument("-5", "--fivefold", default=True, action="store_false", help="run five-fold cross validation (default: True)")
    parser.add_argument("-n", "--name", default=None, help="name of experiment for saving, required")
    parser.add_argument("-t", "--trainfile", default=None, type=str, help="train csv")
    parser.add_argument("-v", "--testfile", default=None, type=str, help="test csv if not five-fold")
    parser.add_argument("-D", "--debug", default=False, action="store_true")

    args = parser.parse_args()
    generate_splits(args)


