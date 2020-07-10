
import time 
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList, Linear,\
        LayerNorm, Sequential, Embedding, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import json
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize

import os

import warnings
warnings.filterwarnings("ignore")


padding_token = '0'
oov_token = '1'
sos_token = '2' # start of sentence, only needed for the target sentence
eos_token = '3' # end of sentence, only needed for the target sentence


class Transformer(Module):
    """
    """
    ARGS = ["N_stacks_encoder", "N_stacks_decoder", "N_heads",
            "dk", "dv","dmodel", "ff_inner_dim", "vocab_source",
            "vocab_target_inv", "max_size", "device","EOS_token",
            "PAD_token", "SOS_token"]
    def __init__(self, N_stacks_encoder, N_stacks_decoder, N_heads, 
                 dk, dv, dmodel, ff_inner_dim, vocab_source,
                 vocab_target_inv, max_size, device, EOS_token=3, 
                 PAD_token=0, SOS_token=2, OOV_token=1):
        """
        Args:
            N_stacks_encoder (int): Number of encoder stacks.
            N_stacks_decoder (int): Number of encoder stacks.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
            vocab_size_source (int)
            vocab_size_target (int)
            max_size (int): maximum size of sentence.
            device (str): "cpu" or "cuda"
            EOS_token (int): Position of the EOS token in vocabulary.
            PAD_token (int): Position of the PAD token in vocabulary.
            SOS_token (int): Position of the SOS token in vocabulary.
        """
        super(Transformer, self).__init__()
        self.max_size = max_size
        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.PAD_token = PAD_token
        self.OOV_token = OOV_token

        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.ff_inner_dim = ff_inner_dim

        self.N_stacks_encoder = N_stacks_encoder
        self.N_stacks_decoder = N_stacks_decoder
        self.N_heads = N_heads

        self.vocab_source = vocab_source
        self.vocab_target_inv = vocab_target_inv

        self.vocab_size_target = np.max(np.array(list(vocab_target_inv.keys()))) + 4
        self.vocab_source_target = np.max(np.array(list(vocab_source.values()))) + 4

        self.source_embedding = Embedding(self.vocab_source_target, dmodel, PAD_token).to(device)
        self.target_embedding = Embedding(self.vocab_size_target, dmodel, PAD_token).to(device)

        self.encoder = Encoder(N_stacks_encoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)

        self.decoder = Decoder(N_stacks_decoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)

        self.output_linear = Linear(dmodel, self.vocab_size_target).to(device)

        self.PE_tensor_x = self.build_encoding(max_size, dmodel).to(device)
        self.PE_tensor_y = self.build_encoding(max_size, dmodel).to(device)

        self.device = device

    def forward(self, x, y=None):
        """ Forward pass of the transformer

        Args:
            x (torch.Tensor): Source sentence, tokenized, (B x T)
            y (torch.Tensor): If defined, teacher forcing is used to process
                rapidly the decoding. (B x T_target)

        Returns:
            (B x T x target_vocab_size) the probability distribution over the taret vocab.
        """
        bs, len_x = x.size()
        
        sos_token = torch.tensor([self.SOS_token]).unsqueeze(0).repeat(bs, 1).to(self.device)
        PE_x = self.PE_tensor_x[:len_x, :].detach().unsqueeze(0).repeat(bs, 1, 1).double()
        PE_x.requires_grad = False

        x = self.source_embedding(x).double()
        encoded = self.encoder(x + PE_x)

        if self.training and not y is None:
            bs, len_y = y.size()
            PE_y = self.PE_tensor_y[:len_y + 1, :].detach().unsqueeze(0).repeat(bs, 1, 1).double()
            PE_y.requires_grad = False

            y = torch.cat([sos_token, y], 1)
            y = self.target_embedding(y).double()
            _, decoded = self.decoder(encoded, y + PE_y)
            out = self.output_linear(decoded)[:, 1:]

        else:
            y = torch.tensor([self.SOS_token] + [self.PAD_token] * self.max_size).unsqueeze(0).repeat(bs, 1).to(self.device)
            PE_y = self.PE_tensor_y[:self.max_size + 1, :].unsqueeze(0).repeat(bs, 1, 1).double()

            for i in range(1, self.max_size + 1):
                embed_y = self.target_embedding(y[:, :i]).double()

                _, decoded = self.decoder(encoded, embed_y + PE_y[:, :i])
                probs = self.output_linear(decoded)

                _, new_y = torch.max(probs, -1)
                y[:, 1:i+1] = new_y
            
            out = y[:, 1:]
        return out

    def fit(self, pairs_train, pairs_test, n_epochs, lr=0.001,
            batch_size=64, seed=42, save_path="./trained_transformer.pt"):
        """ Trains the network

        Args:
            train_loader (torch.utils.data.DataLoader)
            test_loader (torch.utils.data.DataLoader)
            n_epchos (int)
            lr (float)
            batch_size (int)
            seed (int)
            save_path (str)
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(Dataset(pairs_train), batch_size=batch_size, 
                                  shuffle=True, collate_fn=self.my_pad)
        test_loader = DataLoader(Dataset(pairs_test), batch_size=batch_size, 
                              shuffle=True, collate_fn=self.my_pad)

        optimizer = torch.optim.Adam(self.parameters(), lr)
        criteron = CrossEntropyLoss()

        for epoch in range(n_epochs):
            pbar = self.initialize_pbar(epoch, n_epochs, 136521)
            epoch_train_loss = []
            self.train()
            for i, (source, target) in enumerate(train_loader):
                source = source.to(self.device)
                target = target.to(self.device)

                pred = self.forward(source, target)

                optimizer.zero_grad()

                target = target.view(-1)
                loss = criteron(pred.reshape(target.size()[0], -1), target)

                loss.backward()
                optimizer.step()

                pbar.update(source.size(0))
                epoch_train_loss.append(loss.item())
                pbar.set_postfix({"train_loss" : np.mean(epoch_train_loss)})

            test_BLEU = self.test(test_loader, criteron)
            pbar.set_postfix({"train_loss" : np.mean(epoch_train_loss), "test_BLEU": test_BLEU})

        self.save(save_path)

    def test(self, test_loader, criteron):
        """ tests the network on the test_loader 

        Args:
            test_loader (torch.utils.data.DataLoader)
            criteron (torch)
        """
        self.eval()
        BLEU = []
        for i, (source, target) in enumerate(test_loader):
            from nltk.translate.bleu_score import sentence_bleu

            source = source.to(self.device)
            targets = target.to(self.device)
            preds = self.forward(source)
            
            score = 0
            nb_txts = source.size()[0]
            for i in range(nb_txts):
                pred = [k for k in self.targetInts_to_nl(preds[i].squeeze().tolist())]
                tgt = [k for k in self.targetInts_to_nl(targets[i].squeeze().tolist())]
                score += sentence_bleu([tgt], pred) / nb_txts

            BLEU.append(score)
        return np.mean(BLEU)

    def predict(self, sentence):
        """ Translates a sentence """
        sentence = self.sourceNl_to_ints(sentence)
        pred = self.forward(sentence)

        target_ints = pred.squeeze()

        target_nl = self.targetInts_to_nl(target_ints.tolist())
        target_nl = [t for t in target_nl if t not in ['<PAD>', '<SOS>', '<EOS>']]
        return ' '.join(target_nl)

    def save(self, path_to_file):
        """ Saves the architecture and weights of the model 

        Args:
            path_to_file (str)
        """
        attrs = {attr : getattr(self, attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs, path_to_file)

    def my_pad(self, my_list):
        """ my_list is a list of tuples of the form [(tensor_s_1,tensor_t_1),...,(tensor_s_batch,tensor_t_batch)]
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence """
        batch_source = pad_sequence([torch.cat((elt[0],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]
        batch_target = pad_sequence([torch.cat((elt[1],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]

        return batch_source,batch_target 

    def sourceNl_to_ints(self,source_nl):
        '''converts natural language source sentence into source integers'''
        source_nl_clean = source_nl.lower().replace("'",' ').replace('-',' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean, "english")
        source_ints = [int(self.vocab_source[elt]) if elt in self.vocab_source else \
                       self.OOV_token for elt in source_nl_clean_tok] 
        source_ints = torch.LongTensor([source_ints]).to(self.device)

        return source_ints 

    def targetInts_to_nl(self, target_ints):
        '''converts integer target sentence into target natural language'''
        return ['<PAD>' if elt==self.PAD_token else '<OOV>' if elt==self.OOV_token \
                else '<EOS>' if elt==self.EOS_token else '<SOS>' if elt==self.SOS_token\
                else self.vocab_target_inv[elt] for elt in target_ints]

    def shift(self, x):
        """ Adds SOS token to sentence """
        x = torch.cat((torch.ones((x.size(0), 1), dtype=torch.long).to(self.device) * self.SOS_token, x), 1)

        return x

    @classmethod 
    def load(cls, path_to_file, device=None):
        """ Loads a saved model from scrath 

        Args:
            path_to_file (str)
            device (str): If defined, allows to modify to device on which the
                network runs.
        """
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage) 
        state_dict = attrs.pop('state_dict')
        if not device is None:
            attrs["device"] = device

        new = cls(**attrs) 
        new.load_state_dict(state_dict)

        return new

    @staticmethod
    def build_encoding(size, dimension):
        """ Builds a tensor for positional encoding 

        Args:
            size (int): max size of a sentene.
            dimension (int): representation dimentionality in the network.
        """
        pos = torch.arange(size).view(-1,1)
        dim = []
        for _ in range(int(dimension/2)):
            dim.append(np.sin(pos/(10000**(2*_/dimension))))
            dim.append(np.cos(pos/(10000**((2*_+1)/dimension))))

        return torch.cat(dim, 1)

    @staticmethod
    def initialize_pbar(epoch, epochs, its_per_epochs):
        """Initializes a progress bar for the current epoch

        Returns:
            the progess bar (tqdm.tqdm)
        """
        return tqdm(total=its_per_epochs, unit_scale=True,
                    desc="Epoch %i/%i" % (epoch+1, epochs),
                    postfix={})


class Encoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_stacks (int): The number of times the encoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(EncoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Encoder, self).__init__(*stacks)


class EncoderStack(Module):
    """ One stack of encoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(EncoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

        self.layer_norm1 = nn.LayerNorm(dmodel).to(device)
        self.layer_norm2 = nn.LayerNorm(dmodel).to(device)

    def forward(self, x):
        """ Forward pass of one encoder stack

        Args:
            x (torch.tensor): The input sequence (B, T, dmodel)

        Return:
            (torch.tensor) (B, T, dmodel)
        """
        shape = x.size()
        x = x.float()

        att = self.attention(x, x, x)
        x_1 = self.layer_norm1(x + att)

        ff = self.feed_forward(x_1)

        return self.layer_norm2(x_1 + ff)


class Decoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the decoder.

        Args:
            N_stacks (int): The number of times the decoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(DecoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Decoder, self).__init__(*stacks)

    def forward(self, x, y):
        """ Forward pass of the decode
        Needs a modification from the Sequential forward to accept multiple
        inputs.
        """
        for module in self.children():
            x, y = module(x, y)

        return x, y


class DecoderStack(Module):
    """ One stack of decoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the deocder stack.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(DecoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention_1 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.attention_2 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

        self.layer_norm1 = nn.LayerNorm(dmodel).to(device)
        self.layer_norm2 = nn.LayerNorm(dmodel).to(device)
        self.layer_norm3 = nn.LayerNorm(dmodel).to(device)

    def forward(self, x, y):
        """ Forward pass of one decoder stack

        Args:
            x (torch.tensor): The encoded sequence (B, T, dmodel)
            y (torch.tensor): The shifted decoded sequence (B, T, dmodel)
        """
        y = y.float()
        x = x.float()

        att_1 = self.attention_1(y, y, y, maskout=True)
        y_1 = self.layer_norm1(y + att_1)

        att_2 = self.attention_2(y_1, x, x)
        y_2 = self.layer_norm2(y_1 + att_2)

        ff = self.feed_forward(y_2)

        return x, self.layer_norm3(y_2 + ff)


class MultiHeadAttention(Module):
    """ A MultiHeadAttention Module """
    def __init__(self, N_heads, dk, dv, dmodel, device):
        """ Initializes the Multihead

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
        """
        super(MultiHeadAttention, self).__init__()
        self.dk = dk
        self.dv = dv
        self.N_heads = N_heads

        # More efficient and easier to implement
        self.Wq = nn.Linear(dmodel, dk * N_heads, bias=False).to(device)
        self.Wk = nn.Linear(dmodel, dk * N_heads, bias=False).to(device)
        self.Wv = nn.Linear(dmodel, dv * N_heads, bias=False).to(device)
        self.Wq = nn.Linear(dmodel, dmodel, bias=False).to(device)

        self.mask = (1 - torch.triu(torch.ones((1, 100, 100)), diagonal=1)).bool().to(device)

    def attention(self, Q, K, V, maskout=False):
        bs, n_heads, t, d = Q.size()

        QK = torch.matmul(Q, K.transpose(2, 3))
        soft = torch.softmax(QK / np.sqrt(d), dim=-1)
        
        if maskout:
            mask = self.mask[:, :t, :t].view(1, 1, t, t).detach()
            mask.requires_grad = False
            soft = soft.masked_fill(mask == 0, -1e10)

        return torch.matmul(soft, V)

    def forward(self, Q, K, V, maskout=False):
        """ forward of a multihead attention

        Specially implemented to be computed in parallel by pytorch.

        Args:
            Q (torch.Tensor): The queries (B, T, dmodel)
            K (torch.Tensor): The keys (B, T, dmodel)
            V (torch.Tensor): The values (B, T, dmodel)
            maskout (bool): Whether to apply or not forward masking.

        Returns:
            (B, T, dmodel) tensor
        """

        bs, len_q, d = Q.size()
        bs, len_k, d = K.size()
        bs, len_v, d = V.size()

        QWq = self.Wq(Q).view(bs, len_q, self.N_heads, self.dk).transpose(1, 2)
        KWk = self.Wk(K).view(bs, len_k, self.N_heads, self.dk).transpose(1, 2)
        VWv = self.Wv(V).view(bs, len_v, self.N_heads, self.dv).transpose(1, 2)

        heads = self.attention(QWq, KWk, VWv, maskout=maskout)
        heads = heads.transpose(1, 2).contiguous().view(bs, len_q, -1)

        return heads


class FeedForward(Module):
    """ A FeedForward module """
    def __init__(self, inner_dim, dmodel, device):
        """ Initialized the FeedForwrd module

        Args:
            inner_dim (int): Dimension of the hidden layer.
            dmodel (int): The model dimension.
        """
        super(FeedForward, self).__init__()
        self.inner = Linear(dmodel, inner_dim).to(device)
        self.out = Linear(inner_dim, dmodel).to(device)

    def forward(self, x):
        """ Forward of a FeedForward module

        Args:
            x (torch.Tensor)
        """
        return self.out(F.relu(self.inner(x)))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs) # total nb of observations

    def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)


def load_pairs(train_or_test):
    with open("data/" + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()

    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]

    return pairs_tmp


if __name__ == "__main__":

    pairs_train = load_pairs('train')
    pairs_test = load_pairs('test')

    pairs_test = pairs_test #[:100]

    with open("./data/" + 'vocab_source.json','r') as file:
        vocab_source = json.load(file) # word -> index

    with open("./data/" + 'vocab_target.json','r') as file:
        vocab_target = json.load(file) # word -> index

    vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word

    if os.path.exists("trained_transformer.pt"):
        model = Transformer.load("trained_transformer.pt")

    else:
        model = Transformer(N_stacks_encoder=2,
                            N_stacks_decoder=2,
                            N_heads=4, dk=int(256/4), dv=int(256/4), dmodel=256,
                            ff_inner_dim=512,
                            vocab_source=vocab_source,
                            vocab_target_inv=vocab_target_inv,
                            max_size=24, device="cuda")
        model.fit(pairs_train, pairs_test, 10, lr=0.001)


    test_loader = DataLoader(Dataset(pairs_test), batch_size=32, shuffle=False, collate_fn=model.my_pad)
    bleu = model.test(test_loader, 0)

    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
               'This river is full of fish.', # plein vs pleine (accord)
               'The fridge is full of food.', 
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace'] 

    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt, model.predict(elt)))
