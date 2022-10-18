import torch
import torchvision
import esm
import numpy as np
import cv2
from tqdm import tqdm
from PretrainedModels.config import load_model_architecture
from torch.nn.functional import gelu
from LoadingData.tokenizer import Tokenizer
from PIL import Image


class MSARetrieveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esm1b = load_model_architecture(name="esm-1b")

        # 去掉不用的参数的梯度
        for param in self.esm1b.lm_head.parameters():
            param.requires_grad = False
        for param in self.esm1b.contact_head.parameters():
            param.requires_grad = False
            
        self.layer = self.esm1b.num_layers
        self.dim = self.esm1b.args.embed_dim
        
        # for param in self.esm1b.param0eters():
        #     param.requires_grad = False
        #
        # for param in self.esm1b.layers[-1].parameters():
        #     param.requires_grad = True
        
        # self.threshold_model = ThresholdModel(1024, 2048, 1, 1)
        # self.encoder = AutoEncoder()
    
    def forward(self, inputs, lengths=None, **kwargs):
        outputs = self.esm1b(inputs, repr_layers=[self.layer], **kwargs)['representations']
        
        # vec_out = torch.empty_like(outputs[self.layer][:, 0, :])
        if lengths:
            outputs = outputs[self.layer]
            vec_out = [outputs[i, 1: l+1].mean(dim=0) for i, l in enumerate(lengths)]
            vec_out = torch.stack(vec_out, dim=0)
            
        else:
            vec_out = outputs[self.layer][:, 0, :]
            # vec_out = outputs[self.layer].mean(dim=1)
            # print(vec_out.size())
        
        res = {'vec': vec_out}
        
        return res
    
    def get_lm_head(self, inputs):
        return self.esm1b(inputs)['logits']
    
    def get_t(self, inputs, lengths):
        maps = self.get_contact_map(inputs, lengths).unsqueeze(1)
        # import matplotlib.pyplot as plt
        # plt.imshow(maps[0, 0].to('cpu').numpy(), 'gray')
        repr = self.encoder.encode(maps)
        return self.threshold_model(repr)

    def normalize(self, image):
        mean = np.mean(image)
        var = np.mean(np.square(image - mean))

        image = (image - mean) / np.sqrt(var)
        return image
        # vmin = np.min(image)
        # vmax = np.max(image)
        # image = (image - vmin) / (vmax - vmin)
        # # image = np.round(image * 63)
        # return image

    def map_transform(self, map):
        map = cv2.resize(map, (64, 64), interpolation=cv2.INTER_AREA)
        map = self.normalize(map)
        return map

    def get_contact_map(self, inputs, lengths):
        contact_maps = self.esm1b.predict_contacts(inputs, lengths=lengths)
        return contact_maps
        # device = contact_maps[0].device
        #
        # regularized_maps = torch.empty(inputs.size(0), 64, 64, dtype=torch.float).to(device)
        # for i, map in enumerate(contact_maps):
        #     img = self.map_transform(map.to('cpu').numpy())
        #     regularized_maps[i] = torch.tensor(img).to(device)
        #
        # return regularized_maps
        #
    def get_map_repr(self, inputs, lengths):
        maps = self.get_contact_map(inputs, lengths).unsqueeze(1)
        return self.encoder.encode(maps)
    

class ContactMapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # load pretrained resnet50 model
        self.res50 = torchvision.models.resnet50()
        self.res50.load_state_dict(torch.load("/sujin/TwinTowers/PretrainedModels/resnet50-19c8e357.pth"))

    def normalize(self, image):
        mean = torch.mean(image)
        var = torch.mean(np.square(image - mean))

        image = (image - mean) / np.sqrt(var)
        return image
        # vmin = np.min(image)
        # vmax = np.max(image)
    
        # return (image - vmin) / (vmax - vmin)

    def map_transform(self, map):
        # map = cv2.resize(self.normalize(map), (224, 224), interpolation=cv2.INTER_AREA)
        # img = np.round(map * 255)
        img = cv2.resize(map, (64, 64), interpolation=cv2.INTER_AREA)
        return torch.tensor(img)
    
    def forward(self, contact_maps):
        device = contact_maps[0].device
        maps = []
        for i, map in enumerate(contact_maps):
            img = self.map_transform(map.to('cpu').numpy())
            img = torch.tensor(img)[None, None, :, :].repeat(1, 3, 1, 1)
            maps.append(img)
        
        inputs = torch.cat(maps, dim=0).to(device)
        return self.res50(inputs)
    

class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=1024, hidden_layer_num=1, label_num=2):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer_num)])
        self.out = torch.nn.Linear(hidden_size, label_num)

    def forward(self, inputs):
        x = gelu(self.input_layer(inputs))
        
        for layer in self.hidden_layers:
            x = gelu(layer(x))
            
        t = self.out(x)
        return t


class show(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        print(torch.isnan(inputs).sum())
        
        return inputs


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            # input: 1*64*64
            
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=2),  # 32*32*32
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(32, 128, kernel_size=2, stride=2),  # 128*16*16
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
    
            torch.nn.Conv2d(128, 16, kernel_size=2, stride=2),  # 16*8*8
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 128, kernel_size=2, stride=2),  # 128*16*16
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),  # 32*32*32
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # 1*64*64
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(inplace=True),
        )

    def normalize(self, x):
        # mean = torch.mean(x, dim=[-1, -2], keepdim=True)
        # std = torch.std(x, dim=[-1, -2], keepdim=True)
        # image = (x - mean) / std
        # return image
        vmax = x.view(x.size(0), -1).max(dim=1)[0][:, None, None, None]
        vmin = x.view(x.size(0), -1).min(dim=1)[0][:, None, None, None]
        return (x - vmin) / (vmax - vmin)
    
    def symmetrize(self, x):
        return (x + x.transpose(-1, -2)) / 2
    
    def encode(self, inputs):
        repr = self.encoder(inputs).view(inputs.size(0), -1)
        return repr
    
    def forward(self, inputs):
        encode = self.encoder(inputs)
        repr = encode.view(inputs.size(0), -1)
        
        preds = self.decoder(encode)
        symmetrized = self.symmetrize(preds)
        out = self.normalize(symmetrized)
        return out


class MSATransformerRetrieveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msa = load_model_architecture(name="msa")
        self.layer = self.msa.num_layers
        self.dim = self.msa.args.embed_dim

    def forward(self, inputs):
        outputs = self.msa(inputs, repr_layers=[self.layer])['representations']
        vec_out = outputs[self.layer].mean(dim=2)
        res = {'vec': vec_out}
    
        return res


class MutationPredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esm1b = MSARetrieveModel()
        # self.fc = FullyConnectedModel(input_size=self.esm1b.dim,
        #                               hidden_size=self.esm1b.dim*2,
        #                               hidden_layer_num=1,
        #                               label_num=1)
        self.linear = torch.nn.Linear(self.esm1b.dim, self.esm1b.dim)
        self.relu = torch.nn.ReLU()
        self.final = torch.nn.Linear(self.esm1b.dim, 1)

    def forward(self, inputs):
        vec = self.esm1b(inputs)['vec']
        x = self.relu(self.linear(vec))
        x = self.final(x)
        return x
        

class AminoAcidWeightModel(torch.nn.Module):
    def __init__(self, seq, loc, t=1):
        super().__init__()
        tokenizer = Tokenizer()
        gap = 4
        seq_ids = tokenizer.batch_encode([seq])['input_ids'].squeeze(dim=0)
        
        self.aa_weights = torch.zeros(len(loc), 20)
        for i, l in enumerate(loc):
            self.aa_weights[i, seq_ids[l] - gap] = 10
        
        self.aa_weights = torch.nn.Parameter(self.aa_weights, requires_grad=True)
        self.t = t
    
    def forward(self, embeddings):
        # return torch.matmul(self.aa_weights, embeddings)
        return torch.matmul((self.aa_weights / self.t).softmax(dim=-1), embeddings)


class DirectedEvolutionModel(torch.nn.Module):
    def __init__(self, pretrained_model, top_model):
        super().__init__()
        self.esm1b = pretrained_model
        self.top_model = top_model
        
        self.gap = 4
        self.tokenizer = Tokenizer()
        
    def evolve(self, device, seq, loc):
        indices = torch.tensor(range(20)).to(device) + self.gap
        embeddings = self.esm1b.esm1b.embed_tokens(indices).clone().detach()
        
        aa_weights = AminoAcidWeightModel(seq, loc, 1).to(device)
        
        aa_fitness = torch.empty(len(loc), 20)
        with torch.no_grad():
            for time, l in enumerate(loc):
                seq_ids = self.tokenizer.batch_encode([seq])['input_ids'].to(device)
                for i in range(4, 24):
                    seq_ids[:, l] = i
                    out = self.esm1b(seq_ids)
                    vec = out['vec']
                    fitness = self.top_model(vec)
                    aa_fitness[time, i-4] = fitness
        
        optimizer = torch.optim.Adam(aa_weights.parameters(), lr=0.01)
        EPOCH = 1000
        seq_ids = self.tokenizer.batch_encode([seq])['input_ids'].to(device)
        for i in tqdm(range(EPOCH)):
            input_embeddings = self.esm1b.esm1b.embed_tokens(seq_ids)
            loc_embeddings = aa_weights(embeddings).to(device)
            input_embeddings[0, loc] = loc_embeddings
            
            out = self.esm1b(seq_ids, input_embeddings=input_embeddings)
            vec = out['vec']
            fitness = self.top_model(vec)
            
            loss = -fitness
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # print((aa_weights.aa_weights / aa_weights.t).softmax(dim=-1))
        # print(aa_weights.aa_weights.softmax(dim=-1).argsort(descending=True))
        # print(aa_fitness)
        # print(aa_fitness.argsort(descending=True))
        
        # 对每个位点取出权重最大的前3个氨基酸并穷举组合计算fitness
        first_n = 2
        rank = aa_weights.aa_weights.softmax(dim=-1).argsort(descending=True)[:, :first_n]
        test_list = []
        total_num = first_n ** rank.size(0)
        for i in range(total_num):
            id_list = []
            mod_num = i
            for row in range(rank.size(0)):
                id = rank[row, mod_num % first_n].to('cpu').item() + self.gap
                id_list.append(id)
                mod_num = int(mod_num / first_n)
                
            test_list.append(tuple(id_list))
   
        test_fitness = []
        with torch.no_grad():
            seq_ids = self.tokenizer.batch_encode([seq])['input_ids'].to(device)
            for id_list in test_list:
                for i, l in enumerate(loc):
                    seq_ids[:, l] = id_list[i]
                    
                out = self.esm1b(seq_ids)
                vec = out['vec']
                fitness = self.top_model(vec)
                test_fitness.append(fitness.to('cpu').item())

        return test_list, test_fitness
        
    def forward(self, inputs):
        vec = self.esm1b(inputs)['vec']
        fitness = self.top_model(vec)
        
        return fitness
    
    
if __name__ == '__main__':
    model = MSARetrieveModel()
    print(model)
