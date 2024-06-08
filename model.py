from transformers import CLIPModel,CLIPProcessor,BertModel, BertTokenizer,AutoModel,DebertaV2Model,T5ForConditionalGeneration,ViTForImageClassification
from utils import get_dataloader
import time
from tqdm import tqdm
import torch
import numpy as np
import copy
import pdb
import logging
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import os
import math
from open_clip import create_model_from_pretrained
from collections import OrderedDict
import time
# import deepspeed
# from accelerate import Accelerator
# from accelerate.utils import DistributedDataParallelKwargs

puzzle_type_info = pd.read_json("./SMART101-release-v1/puzzle_type_list.jsonl",lines=True)
types = puzzle_type_info["type"].to_list()
values = puzzle_type_info["data"].to_list()
id2type = {} # 每个数据文件夹所对应的问题种类
problem_cnt = len(types)

for j,r in enumerate(values):
    for id in r:
        id2type[id] = j
# # 加载语言模型
# L_model = BertModel.from_pretrained("./lib/bert-base-uncased")
# # L_tokenizer = BertTokenizer.from_pretrained("./lib/bert-base-uncased")
# # 加载视觉模型
# I_model = CLIPModel.from_pretrained("./lib/clip_vit_L_14").vision_model
model,_ = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
img_model = model.visual.trunk
l_model = model.text
def get_log(log_name):
    """
    创建一个日志文件
    :param log_name:日志名
    return logger
    """
    
    logger = logging.getLogger(f"{log_name}")
    logger.setLevel(logging.DEBUG)
    
    path = f"./log/{log_name}.log"
    with open(path, 'w') as file:
        file.write("Hello, World!")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger  

class clip_Model():
    """
    模型
    """
    def __init__(self,args):
        self.args = args
        # 加载模型
        self.model = CLIPModel.from_pretrained("./lib/clip_vit_L_14").to(args.device)
        self.processor = CLIPProcessor.from_pretrained("./lib/clip_vit_L_14")
        # 加载数据集
        self.train_loader = get_dataloader(args, "train")
        self.val_loader = get_dataloader(args, "val")
        # Adam优化器以及cosine计划器
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.lr, betas=(0.9, 0.99))
        # 日志
        self.train_logger = get_log(f"train_log_{args.date}_{args.hyperp_flag}")
        self.eval_logger = get_log(f"eval_log_{args.date}_{args.hyperp_flag}")

    def train_loop(self):
        """
        一轮训练循环
        return 平均loss
        """
        self.model.train()
        tot_loss = 0
        for idx,(img,q_and_a) in tqdm(enumerate(self.train_loader)):
            q_and_a = list(q_and_a)
            inputs = self.processor(text = q_and_a,images = img,return_tensors = "pt",max_length = 77,padding = True,truncation = True)
            inputs = inputs.to(self.args.device)
            # forward
            outputs = self.model(**inputs,return_loss = True)
            loss = outputs.loss
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            tot_loss += loss.item()
        avg_loss = tot_loss / float(idx)
        return avg_loss
    
    def val_loop(self):
        """
        验证
        return acc准确率
        """
        self.train_logger.info("开始测试")
        self.model.eval()
        count_c = 0
        with torch.no_grad():
            for idx,(img,q_and_a,label) in tqdm(enumerate(self.val_loader)):
                # img = img.squeeze(1)
                q_and_a = [item[0] for item in q_and_a]
                inputs = self.processor(text = q_and_a,images = img,return_tensors = "pt",max_length = 77,padding = True,truncation = True) 
                inputs = inputs.to(self.args.device)
                # predict
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                pred = torch.argmax(probs,dim=1)
                
                if pred.item() == label.item():
                    count_c += 1
        lenth = len(self.val_loader)
        acc = float(count_c) / float(lenth)
        return acc
        
    def train(self):
        """
        训练模型
        """
        self.train_logger.info(f"开始训练,batch_size:{self.args.batch_size},lr:{self.args.lr},patience:{self.args.patience}")
        best_acc = 0
        no_improvement = 0 # 超过几轮没有提升则早停
        improv_flag = ""
        patience = self.args.patience
        for i in range(self.args.num_epochs):
            tt = time.time()
            loss = self.train_loop()
            tt = time.time() - tt
            
            acc = self.val_loop()
            improv_flag = ""
            if acc > best_acc:
                best_acc = acc
                model_state_dict = self.model.state_dict()
                path = f"{self.args.save_path}ckpt_{self.args.date}_{self.args.hyperp_flag}.pth"
                torch.save(model_state_dict,path)
                improv_flag = "*"
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > patience:
                    self.train_logger.info(f"超过{patience}轮没有提升，早停")
                    break
            self.train_logger.info(f"{improv_flag}epoch:{i},train_loss:{loss},acc:{acc},train_time:{tt}")

class FeedForwardNet(nn.Module):
    # 前馈神经网络层
    def __init__(self,d_model):
        super(FeedForwardNet,self).__init__()
        self.d_ffn = 4096 # 隐藏层维度
        self.d_model = d_model
        self.ff = nn.Sequential(nn.Linear(d_model,self.d_ffn,bias = False),
                                nn.ReLU(),
                                nn.Linear(self.d_ffn,d_model,bias = False)) # 全连接三层神经网络
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self,x):
        residual = x
        x = self.ff(x)
        x = self.layernorm(x + residual)
        return x

class b_multiheadattention(torch.nn.Module):
    # 多头注意力层
    def __init__(self):
        super(b_multiheadattention,self).__init__()
        d_I_model = 1024
        d_L_model = 768
        self.d_q,self.d_k,self.d_v= 1024,1024,1024
        self.num_heads = 8
        self.W_q_I = torch.nn.Linear(d_I_model,self.d_q * self.num_heads,bias=False) 
        self.W_k_L = torch.nn.Linear(d_L_model,self.d_k * self.num_heads,bias=False)
        self.W_v_L = torch.nn.Linear(d_L_model,self.d_v * self.num_heads,bias=False)
        self.W_O = torch.nn.Linear(self.d_v * self.num_heads,1024,bias=False)
        self.layernorm = nn.LayerNorm(1024) # 正则化
        self.sf = torch.nn.Softmax(dim = -1)
    
    def forward(self,img_embed,sent_embed,atten_mask):
        batch_size = img_embed.size(0)
        residual = img_embed
        q_I = self.W_q_I(img_embed)
        k_L = self.W_k_L(sent_embed)
        v_L = self.W_v_L(sent_embed)
        
        Q = torch.reshape(q_I,(batch_size,-1,self.num_heads,self.d_q)).transpose(1,2) # (batch_size,num_heads,len_s,d_q)
        K = torch.reshape(k_L,(batch_size,-1,self.num_heads,self.d_k)).transpose(1,2) # (batch_size,num_heads,len_s,d_k)
        V = torch.reshape(v_L,(batch_size,-1,self.num_heads,self.d_v)).transpose(1,2) # (batch_size,num_heads,len_s,d_v)
        
        score = torch.matmul(Q,K.transpose(2,-1)) / math.sqrt(K.size(-1))
        # 生成pad mask
        atten_mask = atten_mask.unsqueeze(1) # 扩展一个中间维度
        atten_mask = atten_mask.unsqueeze(1)
        atten_mask = atten_mask.repeat(1,score.size(1),score.size(2),1)
        atten_mask = (atten_mask != 0)
        
        score = score.masked_fill(~atten_mask,value=float('-inf'))
        score_sf = self.sf(score)
        score = torch.matmul(score_sf,V).transpose(1,2)
        score = torch.reshape(score,(batch_size,-1,self.num_heads * self.d_v))
        score = self.W_O(score)
        atten = score + residual
        atten_LN = self.layernorm(atten) 
        
        return atten_LN
    
class self_attention(torch.nn.Module):
    # 多头注意力层
    def __init__(self):
        super(self_attention,self).__init__()
        d_I_model = 1024
        d_L_model = 768
        self.d_q,self.d_k,self.d_v= 1024,1024,1024
        self.num_heads = 8
        self.linear = torch.nn.Linear(768,1024,bias=False)
        self.W_q = torch.nn.Linear(1024,self.d_q * self.num_heads,bias=False) 
        self.W_k = torch.nn.Linear(1024,self.d_k * self.num_heads,bias=False)
        self.W_v = torch.nn.Linear(1024,self.d_v * self.num_heads,bias=False)
        self.W_O = torch.nn.Linear(self.d_v * self.num_heads,1024,bias=False)
        self.layernorm = nn.LayerNorm(1024) # 正则化
        self.sf = torch.nn.Softmax(dim = -1) 
          
    def forward(self,VL_embed,atten_mask):
        batch_size = VL_embed.size(0)
        residual = VL_embed
        
        Q = self.W_q(VL_embed)
        K = self.W_k(VL_embed)
        V = self.W_v(VL_embed)
        
        Q = torch.reshape(Q,(batch_size,-1,self.num_heads,self.d_q)).transpose(1,2) # (batch_size,num_heads,len_s,d_q)
        K = torch.reshape(K,(batch_size,-1,self.num_heads,self.d_k)).transpose(1,2) # (batch_size,num_heads,len_s,d_k)
        V = torch.reshape(V,(batch_size,-1,self.num_heads,self.d_v)).transpose(1,2) # (batch_size,num_heads,len_s,d_v)
        
        score = torch.matmul(Q,K.transpose(2,-1)) / math.sqrt(K.size(-1))
        # 生成padmask
        atten_mask_plus = torch.ones((batch_size,257)).to(atten_mask.device)
        atten_mask = torch.cat((atten_mask_plus,atten_mask),dim=1)
        atten_mask = atten_mask.unsqueeze(1) # 扩展一个中间维度
        atten_mask = atten_mask.unsqueeze(1)
        atten_mask = atten_mask.repeat(1,score.size(1),score.size(2),1)
        atten_mask_T = atten_mask.transpose(2,3)
        atten_mask = (atten_mask + atten_mask_T == 2)
        
        score = score.masked_fill(~atten_mask,value=float('-inf'))
        score_sf = self.sf(score)
        score = torch.matmul(score_sf,V).transpose(1,2)
        score = torch.reshape(score,(batch_size,-1,self.num_heads * self.d_v))
        score = self.W_O(score)
        atten = score + residual
        atten_LN = self.layernorm(atten) 

        return atten_LN
     
class VL_bridge_encoder_layer(torch.nn.Module):
    def __init__(self,repre_dim):
        super(VL_bridge_encoder_layer,self).__init__()
        # # 桥接多头注意力
        # self.attention_layer = b_multiheadattention()
        self.repre_dim = repre_dim
        self.attention_layer = torch.nn.MultiheadAttention(self.repre_dim,8,batch_first=True)
        self.ffn = FeedForwardNet(self.repre_dim)
    
    def forward(self,img_LN,sent_embed,atten_mask):
        """连接器的前向传播

        Args:
            img (batch_size * num_patches * d_I_model): 图像
            sent (batch_size * seq_lenth * d_L_model): 文本

        Returns:
            tensor: 表征
        """
        atten_mask = (atten_mask == 0)
        atten_output,_ = self.attention_layer.forward(query = img_LN,key = sent_embed,value = sent_embed,key_padding_mask = atten_mask)
        hidden_state = self.ffn(atten_output) 
        return hidden_state

class VL_self_encoder_layer(torch.nn.Module):
    def __init__(self):
        super(VL_self_encoder_layer,self).__init__()
        # # 跨模态自注意力
        # self.attention_layer = self_attention()
        self.attention_layer = torch.nn.MultiheadAttention(1024,8,batch_first=True)
        self.ffn  = FeedForwardNet(1024)
    
    def forward(self,VL_embed,atten_mask):
        batch_size = VL_embed.size(0)
        # 生成padmask
        atten_mask_plus = torch.ones((batch_size,257)).to(atten_mask.device)
        atten_mask = torch.cat((atten_mask_plus,atten_mask),dim=1)
        atten_mask = (atten_mask == 0)
        atten_output,_ = self.attention_layer.forward(query = VL_embed,key = VL_embed,value = VL_embed,key_padding_mask=atten_mask)
        hidden_state = self.ffn(atten_output)
        return hidden_state

class VL_encoders(torch.nn.Module):
    def __init__(self,cnt_layer):
        super(VL_encoders,self).__init__()
        # 语义空间映射到视觉空间
        self.L_toI_linear = torch.nn.Linear(768,1024,bias=False)
        # 加载语言embedding
        self.L_embeding = L_model.embeddings
        # 加载视觉embedding
        self.I_embedding = I_model.embeddings
        self.I_pre_layernorm = I_model.pre_layrnorm 
        
        self.encoders = torch.nn.ModuleList([VL_bridge_encoder_layer() for i in range(cnt_layer)]) # 桥接注意力
        # self.encoders = torch.nn.ModuleList([VL_self_encoder_layer() for _ in range(cnt_layer)]) # 跨模态自注意力
        # pooler
        self.pooler_linear = torch.nn.Linear(1024,1024,bias=True)
        self.activate = torch.nn.Tanh()
        # 线性分类层
        self.linear = torch.nn.Linear(1024,5)
        
    def forward(self,img,token_ids,atten_mask):
        # 图像嵌入、文本嵌入
        img_embed = self.I_embedding(img)
        img_LN = self.I_pre_layernorm(img_embed)
        sent_embed = self.L_embeding(token_ids)
        
        # 语义空间向视觉空间映射
        L_toI_embed = self.L_toI_linear(sent_embed)
        # V_concat_L = torch.cat((img_LN,L_toI_embed),dim=1)    
        for layer in self.encoders:
            img_LN = layer.forward(img_LN,L_toI_embed,atten_mask)
            # V_concat_L = layer.forward(V_concat_L,atten_mask)
            
        hidden_state = img_LN
        # hidden_state = V_concat_L
        pooler_output = self.activate(self.pooler_linear(hidden_state[:,0,:]))
        logits = self.linear(pooler_output)
        
        return hidden_state,logits

class VL_fusion_encoders(torch.nn.Module):
    def __init__(self,cnt_layer):
        super(VL_fusion_encoders,self).__init__()
        self.repre_dim = 1152
        # 加载语言模型
        self.L_model = AutoModel.from_pretrained("./lib/deberta-v3-large/deberta-v3-large")
        # # L_tokenizer = BertTokenizer.from_pretrained("./lib/bert-base-uncased")
        # # 加载视觉模型
        # self.I_model = CLIPModel.from_pretrained("./lib/clip_vit_L_14").vision_model
        # self.I_model = CLIPModel.from_pretrained("./lib/clip_vit_L_336_14/clip-vit-large-patch14-336").vision_model
        # 语义空间映射到视觉空间
        self.L_toI_linear = torch.nn.Linear(1024,self.repre_dim,bias=False)
        # # 视觉空间向语义空间映射
        # self.I_toL_linear = torch.nn.Linear(1024,768,bias=False)
        
        self.I_model = torch.nn.Sequential(img_model.patch_embed,img_model.pos_drop,
                                           img_model.patch_drop,img_model.norm_pre,
                                           img_model.blocks,img_model.norm)
        # self.L_model = torch.nn.Sequential(l_model.token_embedding,l_model.transformer,
        #                                    l_model.ln_final,l_model.text_projection)
        # self.I_model = ViTForImageClassification.from_pretrained("./lib/ViT_b_16/ViT-B-16-224")
        #fusion模块,桥接自注意力
        self.fusion = torch.nn.ModuleList([VL_bridge_encoder_layer(self.repre_dim) for i in range(cnt_layer)])
        # pooler
        # self.pooler_linear = torch.nn.Linear(self.repre_dim,self.repre_dim,bias=True)
        # self.activate = torch.nn.Tanh()
        self.pooler = torch.nn.Sequential(img_model.attn_pool,img_model.fc_norm,
                                          img_model.head_drop,img_model.head)
        # 线性分类层
        self.linear = torch.nn.Linear(self.repre_dim,5)
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self,img,token_ids,atten_mask):
        # img_repre = self.I_model(img).last_hidden_state
        text_repre = self.L_model(token_ids,attention_mask = atten_mask).last_hidden_state
        img_repre = self.I_model(img)
        # img_repre = self.I_model(**img)
        # text_repre = self.L_model(token_ids)
        # 语义空间向视觉空间映射
        L_toI = self.L_toI_linear(text_repre)
        # # 视觉空间向语义空间映射
        # I_toL = self.I_toL_linear(img_repre)
        for layer in self.fusion:
            img_repre = layer.forward(img_repre,L_toI,atten_mask)
            # img_repre = layer.forward(img_repre,text_repre,atten_mask)
            # I_toL = layer.forward(I_toL,text_repre,atten_mask)
        # hidden_state = img_repre
        # hidden_state = I_toL
        # pooler_output = self.activate(self.pooler_linear(hidden_state[:,0,:]))
        pooler_output = self.pooler(img_repre)
        logits = self.linear(pooler_output)
        # if mode == "train":
        #     loss = self.criterion(logits,target)
        # else:
        #     probs = logits.softmax(dim=1)
        #     pred = torch.argmax(probs,dim=-1)
        #     loss = torch.sum(pred == target)
        
        return logits
        
class VL_fusion_t5(torch.nn.Module):
    def __init__(self,cnt_layer):
        super(VL_fusion_t5,self).__init__()
        pdb.set_trace()
        # 加载语言模型
        L_model = T5ForConditionalGeneration.from_pretrained("./lib/flant5-xxl/flan_t5_xxl")
        # 语言编码器与解码器
        self.L_encoder = L_model.encoder
        self.L_decoder = L_model.decoder
        # 加载视觉模型
        self.I_model = CLIPModel.from_pretrained("./lib/clip_vit_L_14").vision_model
        # 融合模块
        self.fusion = torch.nn.ModuleList([VL_bridge_encoder_layer(self.repre_dim) for i in range(cnt_layer)])
    
    def forward(self,img,token_ids,atten_mask):
        img_repre = self.I_model(img).last_hidden_state
        
class bridge_atten_Model():
    """
    """
    def __init__(self,args):
        self.device_ids = [int(idx) for idx in args.device.split(",")]
        self.args = args
        # 加载数据集
        # self.train_loaders = {}
        # for t in types:
        # # t = types[3]
        # self.train_loaders[t] = get_dataloader(args,"train")
        # self.val_loaders = {}
        # for t in types:
        # self.val_loaders[t] = get_dataloader(args,"val") 
        self.train_loader = get_dataloader(args,"train")
        # self.val_loader = get_dataloader(args,"val")
        # 日志
        self.train_logger = get_log(f"train_log_{args.date}_{args.hyperp_flag}")
        # # 编码器
        # self.encoders = dict(zip(types,[self.move_toGPUs() for i in range(problem_cnt)]))
        self.encoder = self.move_toGPUs()
        # 优化器
        # self.optimizers = dict(zip(types,[torch.optim.Adam(self.encoders[tp].parameters(),lr=args.lr, betas=(0.9, 0.99)) for tp in types]))
        self.optimizer = torch.optim.Adam(self.encoder.parameters(),lr=args.lr) 
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def move_toGPUs(self):
        """将atten模块移动到GPU上
        Returns:
            _type_: _description_
        """
        # vl_encoder = VL_encoder_layer()
        # vl_encoder = VL_encoders(self.args.cnt_layer)
        vl_encoder = VL_fusion_encoders(self.args.cnt_layer)
        # vl_encoder = VL_fusion_t5(self.args.cnt_layer)
        vl_encoder = nn.DataParallel(vl_encoder,device_ids=self.device_ids).to("cuda") 
        return vl_encoder

    def train_loop(self):
        """
        一轮训练循环
        return 平均loss
        """
        # self.L_model.train()
        tot_loss = 0
        cnt = 0
        # for (tp,train_loader) in tqdm(self.train_loaders.items()):
        #     self.encoders[tp].train()
        for img,token_ids,atten_mask,target in tqdm(self.train_loader):
            # 得到表征
            # _,logits = self.encoders[tp].forward(img,token_ids,atten_mask)
            logits = self.encoder.forward(img,token_ids,atten_mask)
            # loss = loss.sum()/len(self.device_ids)
            
            # 计算损失
            target = target.to(logits.device)
            loss = self.criterion(logits,target)
            # # backward
            # self.optimizers[tp].zero_grad()
            # loss.backward()
            # self.optimizers[tp].step()
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            tot_loss += loss.item()
            cnt += 1
        avg_loss = tot_loss / float(cnt)
        return avg_loss
     
    def train(self):
        """
        训练模型
        """
        self.train_logger.info(f"开始训练,batch_size:{self.args.batch_size},lr:{self.args.lr},patience:{self.args.patience}")
        # best_acc = 0
        best_loss = 99999999
        no_improvement = 0 # 超过几轮没有提升则早停
        improv_flag = ""
        patience = self.args.patience
        for i in range(self.args.num_epochs):
            tt = time.time()
            loss = self.train_loop()
            tt = time.time() - tt

            # acc = self.val_loop()
            acc = None
        
            improv_flag = ""
            # if acc > best_acc:
                # best_acc = acc
            if loss < best_loss:
                best_loss = loss
                # for tp in types:
                #     model_state_dict = self.encoders[tp].state_dict()
                #     path = f"{self.args.save_path}ckpt_{self.args.date}_{self.args.hyperp_flag}_{tp}.pth"
                #     torch.save(model_state_dict,path)
                model_state_dict = self.encoder.state_dict()
                path = f"{self.args.save_path}ckpt_{self.args.date}_{self.args.hyperp_flag}.pth"
                torch.save(model_state_dict,path)
                improv_flag = "*"
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    self.train_logger.info(f"超过{patience}轮没有提升，早停")
                    break
            self.train_logger.info(f"{improv_flag}epoch:{i},train_loss:{loss},acc:{acc},train_time:{tt}")
    
    def val_loop(self):
        """
        验证
        return acc准确率
        """
        self.train_logger.info("开始测试")
        # self.model.eval()
        count_c = 0
        cnt = 0
        with torch.no_grad():
            # for (tp,val_loader) in tqdm(self.val_loaders.items()):
            #     self.encoders[tp].eval()
            for img,token_ids,atten_mask,label in tqdm(self.val_loader):
                logits = self.encoder(img,token_ids,atten_mask)
                probs = logits.softmax(dim=1)
                pred = torch.argmax(probs,dim=-1)
                label = label.to(pred.device)
                count_c += torch.sum(pred == label).item()
                # count_c += cnt_c.sum()
            cnt += len(self.val_loader) * self.args.batch_size
        acc = float(count_c) / float(cnt)
        return acc

class deepspeed_trainModel():
    def __init__(self,args):
        model = VL_fusion_encoders(args.cnt_layer)
        train_dataset = get_dataloader(args,"train")
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        deepspeed.init_distributed()
        self.args = args
        # 模型，优化器，计划器，训练集和验证集
        self.model, self.optimizer,self.train_dataloader,self.scheduler = deepspeed.initialize(args=args,model=model,optimizer=optimizer,model_parameters=model.parameters(),training_data=train_dataset,lr_scheduler=scheduler)
        self.val_dataloader = get_dataloader(args,"val")
        # 日志
        self.train_logger = get_log(f"train_log_{args.date}_{args.hyperp_flag}")
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_loop(self):
        """
        一轮训练循环
        return 平均loss
        """
        tot_loss = 0
        cnt = 0
        for step,(index,img,token_ids,atten_mask,target) in enumerate(self.train_dataloader):
            img,token_ids,atten_mask,target = img.to(self.model.device),token_ids.to(self.model.device),atten_mask.to(self.model.device),target.to(self.model.device)
            # 前向传播
            logits = self.model(img,token_ids,atten_mask)
            loss = self.criterion(logits,target)
            #runs backpropagation
            self.model.backward(loss)

            #weight update
            self.model.step()
            tot_loss += loss.item()
            cnt += 1
        avg_loss = tot_loss / float(cnt)
        return avg_loss
     
    def train(self):
        """
        训练模型
        """
        self.train_logger.info(f"开始训练,batch_size:{self.args.batch_size},lr:{self.args.lr},patience:{self.args.patience}")
        best_acc = 0
        no_improvement = 0 # 超过几轮没有提升则早停
        improv_flag = ""
        patience = self.args.patience
        for i in range(self.args.num_epochs):
            tt = time.time()
            loss = self.train_loop()
            tt = time.time() - tt

            acc = self.val_loop()
            improv_flag = ""
            if acc > best_acc and self.model.local_rank==0:
                best_acc = acc
                # save model
                # path = f"{self.args.save_path}ckpt_{self.args.date}_{self.args.hyperp_flag}.pth"
                client_state = {
                                    "epoch": i,
                                    "validation_accuracy": acc
                                }
                self.model.save_checkpoint(save_dir=self.args.save_path,tag=acc,client_state=client_state)
                improv_flag = "*"
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    self.train_logger.info(f"超过{patience}轮没有提升，早停")
                    break
            self.scheduler.step()
            self.train_logger.info(f"{improv_flag}epoch:{i},train_loss:{loss},acc:{acc},train_time:{tt}")
    
    def val_loop(self):
        """
        验证
        return acc准确率
        """
        self.train_logger.info("开始测试")
        count_c = 0
        cnt = 0
        with torch.no_grad():
            for index,img,token_ids,atten_mask,label in self.val_dataloader:
                logits = self.model(img,token_ids,atten_mask)
                probs = logits.softmax(dim=1)
                pred = torch.argmax(probs,dim=-1)
                count_c += torch.sum(pred == label).item()
            cnt += len(self.val_loader) * self.args.batch_size
        acc = float(count_c) / float(cnt)
        return acc      
