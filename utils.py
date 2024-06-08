from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import pandas as pd
import torch
from PIL import Image
import pdb
from transformers import AutoTokenizer,DebertaV2Tokenizer,T5Tokenizer,ViTImageProcessor
from torch.utils.data.distributed import DistributedSampler
from open_clip import create_model_from_pretrained,get_tokenizer

# 初始化PS分割方式中验证集以及训练集的索引列表
# PS_VAL_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77,18] # 多加一个18（order）
# PS_TRAIN_IDX = []
PS_TRAIN_IDX = [i+1 for i in range(101)]
# for i in range(101):
#     if i + 1 not in PS_VAL_IDX:
#         PS_TRAIN_IDX.append(i+1)

# # 题目类别,将数据集分割成8种问题的训练集再训练
# puzzle_type_list = pd.read_json("./SMART101-release-v1/puzzle_type_list.jsonl",lines=True)
# types = puzzle_type_list["type"]
# data = puzzle_type_list["data"]
# puzzle_dict_train = dict.fromkeys(types)
# puzzle_dict_val = dict.fromkeys(types)
# for idx,d in enumerate(data):
#     t = []
#     v = []
#     for ID in d:
#         if ID in PS_VAL_IDX:
#             v.append(ID)
#         else:
#             t.append(ID)
#     puzzle_dict_train[types[idx]] = t
#     puzzle_dict_val[types[idx]] = v
# 分词器 
L_tokenizer = AutoTokenizer.from_pretrained("./lib/deberta-v3-large/deberta-v3-large")
# L_tokenizer = T5Tokenizer.from_pretrained("./lib/flant5-xxl/flan_t5_xxl")
# L_tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')

def concat_dataframe(data_root,idx_list):
    """
    拼接dataframe
    :param idx_list:索引列表
    return [dataframe]
    """
    data_list = []
    for idx in idx_list:
        path = f"{data_root}{idx}/puzzle_{idx}.csv"
        data_i = pd.read_csv(path,index_col=False)
        data_list.append(data_i) 
    data = pd.concat(data_list,axis=0,ignore_index=True)
    data = data.sample(len(data),ignore_index=True) # 对数据进行一个随机采样，打乱顺序
    return data

def Construct_data_info(data_root,mode,tp):
    """
    读取数据信息的csv文件
    :param mode:模式，是训练还是测试
    return Dataframe
    """
    if tp != None:
        if mode == "train":
            return concat_dataframe(data_root,puzzle_dict_train[tp])
        else:
            return concat_dataframe(data_root,puzzle_dict_val[tp])
    else:
        if mode == "train":
            return concat_dataframe(data_root,PS_TRAIN_IDX)
        else:
            return concat_dataframe(data_root,PS_VAL_IDX)
        
class SMART_dataset(Dataset):
    """
    SMART数据集
    """
    def __init__(self,args,mode,tp = None):
        super().__init__()
        self.mode = mode
        self.data_root = args.data_root
        self.data_info = Construct_data_info(args.data_root,mode,tp) # 数据信息
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path) # 加载LLM的分词器
        # self.transform = Compose(
        #         [
        #             Resize(224),  # if the images are of higher resolution. we work with pre-resized 224x224 images.
        #             # RandomCrop(224),
        #             ToTensor(),
        #             # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        #         ]
        #     ) # 图像分辨率转化器
        _,self.transform = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # self.transform = ViTImageProcessor.from_pretrained("./lib/ViT_b_16/ViT-B-16-224")
        
    def transform_img(self,img_path):
        """
        读取图像，并做分辨率的处理
        :param img_path:图像路径
        return 转化后的图像
        """
        im = Image.open(img_path).convert("RGB")
        return self.transform(im)
        # return self.transform(images=im,return_tensors="pt")
        # return im
    
    def __getitem__(self, idx):
        """
        返回数据
        :param idx:数据索引
        return Image,question+answer
        """
        question = self.data_info.loc[idx,"Question"]
        img_name = self.data_info.loc[idx,"image"]
        # 选项
        options = {}
        for opt in ["A","B","C","D","E"]:
            options[opt] = self.data_info.loc[idx,opt]
        label = self.data_info.loc[idx,"Answer"]
        answer = options[label]
        # 图像路径
        img_name_split = img_name.split("_")
        index = img_name_split[1]
        # 数据集中61\62\64文件夹出现问题
        if index == '104':
            index = 64
        if index == '102':
            index = 61
        if index == "103":
            index = 62
        img_path = f"{self.data_root}{index}/img/{img_name}"
        if self.mode == "train":
            """
            如果是训练集，那么直接将问题和答案连接做训练
            """
            img = self.transform_img(img_path)
            # 将问题和答案连接
            q = f"{question} + A {options['A']} B {options['B']} C {options['C']} D {options['D']} E {options['E']}"
            target = ord(label) - ord("A")
            tokens = L_tokenizer(q,padding = "max_length",truncation = True,max_length = 96,return_tensors="pt")
            # tokens = L_tokenizer(q,padding = "longest",truncation = True,return_tensors="pt")
            # token_ids = L_tokenizer(q,context_length=64)
            token_ids = tokens["input_ids"]
            atten_mask = tokens["attention_mask"]
            # atten_mask = (token_ids != 1).int()
            token_ids = torch.squeeze(token_ids,dim=0)
            atten_mask = torch.squeeze(atten_mask,dim=0)
            # img["pixel_values"] = torch.squeeze(img["pixel_values"],dim=0)
            return img,token_ids,atten_mask,target
        else:
            """
            如果是验证集，将同一个问题的多个选项做连接
            """
            img = self.transform_img(img_path)
            # # 将图像copy5份然后写入同一个tensor
            # c,h,w = img.size(0),img.size(1),img.size(2)
            # img_copies = torch.zeros([5,c,h,w],dtype=img.dtype)
            # for i in range(5):
            #     img_copies[i,:,:,:] = img[:,:,:]
            # 将问题与答案连接
            # q_and_a = []
            # for opt in ["A","B","C","D","E"]:
            #     opt_ans = options[opt]
            #     q_and_opt = question + str(opt_ans)
            #     q_and_a.append(q_and_opt)
            q = f"{question} + A {options['A']} B {options['B']} C {options['C']} D {options['D']} E {options['E']}"
            label = ord(label) - ord("A")
            tokens = L_tokenizer(q,padding = "max_length",truncation = True,max_length = 96,return_tensors="pt")
            # tokens = L_tokenizer(q,padding = "longest",truncation = True,return_tensors="pt")
            token_ids = tokens["input_ids"]
            atten_mask = tokens["attention_mask"]
            # token_ids = L_tokenizer(q,context_length=64)
            # token_ids = torch.squeeze(token_ids,dim=0)
            # atten_mask = torch.squeeze(atten_mask,dim=0) 
            # atten_mask = (token_ids != 1).int()
            token_ids = torch.squeeze(token_ids,dim=0)
            atten_mask = torch.squeeze(atten_mask,dim=0)
            # img["pixel_values"] = torch.squeeze(img["pixel_values"],dim=0)
            return img,token_ids,atten_mask,label
        
    def __len__(self):
        return len(self.data_info)

def get_dataloader(args,mode,tp = None):
    
    """
    获取数据加载器
    :param args
    :param mode:模式
    return Dataloader
    """
    smart_dataset = SMART_dataset(args,mode,tp)
    if args.method == "Deepspeed":
        if mode == "train":
            return smart_dataset
        else:
            validation_sampler = DistributedSampler(smart_dataset, shuffle=False)
            data_loader = DataLoader(dataset=smart_dataset, batch_size=args.batch_size, sampler=validation_sampler)
    elif args.method == "clip":
        if mode == "train":
            data_loader = DataLoader(smart_dataset,batch_size = args.batch_size,shuffle=False)
        else:
            data_loader = DataLoader(smart_dataset,batch_size = 1,shuffle=False)
    elif args.method == "VL_encoder":
        data_loader = DataLoader(smart_dataset,batch_size = args.batch_size,shuffle=False)
    return data_loader