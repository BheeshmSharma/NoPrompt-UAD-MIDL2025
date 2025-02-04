from audioop import tomono
import copy
import math
from os import TMP_MAX
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from metrics.metrics import eval_metrics

from torchvision import transforms
from PIL import Image

import numpy as np
from tqdm import tqdm

import shutil
import os

import cv2

from utils.utils import create_directory, setup_folders, calculate_batch_metrics, write_results_to_file,get_file_names, process_image, save_mask


import json

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",

    #Brain dataset that I have just added
    "BraTs20": "a photo of a {} brain.",
    "BraTs20b": "a photo of a {} brain.",
    "BraTs20c": "a photo of a {} brain.",
    
    "BraTs21": "a photo of a {} brain.",
    "BraTs21b": "a photo of a {} brain.",
    "BraTs21c": "a photo of a {} brain.",
    
    "MSLUB": "a photo of a {} brain.",
    "ATLAS": "a photo of a {} brain.",
    "BraTS23": "a photo of a {} brain.",
    "MSD": "a photo of a {} brain.",
}

# ================================================================= #
# 这个是VPT DEEP+CoOp（Frozen）+CLIP（Frozen）的实现。                #
# ================================================================= #


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  Pure CoOp!                  #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        #prompts->[100,77,512] (these are what have the learnable ctx vectors)
        #tokenized_prompts ->[100,77] -> now these are the prompts before learners and have XXXX
       
        x = prompts + self.positional_embedding.type(self.dtype)
        #x -> [100,77,512]
       

        x=x.permute(1,0,2)
        x=self.transformer(x)
       
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        #x->[100,77,512]

        # x.shape = [batch_size, n_ctx, transformer.width] hmmmm what??

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #so in this step kinda like were taking only a few features after doing the linear layer, and then passing through the projection layer

        #x-> [100,512]

        '''IN SHORT WHAT HAPPENS IN THIS TEXT ENCODER IS THAT THE PROMPTS (LEARNABLE) ARE PASSED THROUGH THE CLIP TEXT TRANSFORMER, AND WE CHOOSE
        ONLY A PART OF THE OUTPUT, BASED ON THE TOKENIZED PROMPTS (TOKENS OF 'X X X X X CLS.') , WHICH WE THEN PASS THROUGH THE PROJECTION LAYER'''
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, label=None, devices=None):
        super().__init__()
        if(label):
            n_cls=1
            classnames=[label]
            print('PROMPT LEARNER', classnames)
        else:
            n_cls = len(classnames)                             #p self explanatory
        n_ctx = cfg.TRAINER.COOP.N_CTX                      #'M' in the paper
        ctx_init = cfg.TRAINER.COOP.CTX_INIT                # Going to be ' ' in our case
        # tem_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tem_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]       #well see soon what this does
        clip_imsize = clip_model.visual.input_resolution    #image input (224)
        cfg_imsize = cfg.INPUT.SIZE[0]                      #224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        #[NOT GOING TO BE OUR CASE]
        if ctx_init:
            # use given words to initialize context vectors 
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        #NOPE AGAIN
        elif tem_init:     
            target_nctx = n_ctx
            ctx_init = tem_init[:-1]
            self.class_token_position = "template"
            ctx_init = ctx_init.replace("_", " ")
            ctx_init = ctx_init.split(' ')
            if "{}" in ctx_init:
                self.cls_loc= ctx_init.index("{}")
                ctx_init.remove("{}")
            elif "{}," in ctx_init:
                self.cls_loc= ctx_init.index("{},")
                ctx_init.remove("{},")
            elif "{}." in ctx_init:
                self.cls_loc= ctx_init.index("{}.")
                ctx_init.remove("{}.")
            n_ctx = len(ctx_init)
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            tmp = torch.empty(target_nctx-n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(tmp, std=0.02)
            ctx_vectors = torch.cat([embedding[0, 1 : 1 + n_ctx, :], tmp], dim=0)
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = " ".join(ctx_init)+" "+" ".join(["X"]*(target_nctx-n_ctx))
            n_ctx = target_nctx

        # random initialization  [THHIS OVER HERE IS THE REAL DEAL SIR]
        else:
           

            #NOW WE WOULD DO THIS IF WE CLASS SPECIFIC CONTEXT, WHICH IS NOT THE CASE FOR US, SO WE MOVE AHEAD]
            if cfg.TRAINER.COOP.CSC:            
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            #HERE IS FOR THE UNIFIED CONTEXT
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                #ctx vectors -> [16,512]
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            #prompt_prefix -> only gonna be 16 X followed by _ in the string

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        #so this part has nothing to do with the classnames we set till now, its completely randomly initialized

        classnames = [name.replace("_", " ") for name in classnames]
        #we have 100 classnames like face, leopard etc
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #not used here, but just a list of the length of the encodings of each name
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        #so basically its like a big list of items, wheres these items are basically those 16 Xs followed by the class and then '.'
        #something like ['X X X X X X X X X X X X X X X X face.'] and 100 elements like this
        

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(devices)
        #so basically just tokenizes the above prompts by doing this
            #-> adds a <SOS> token first
            #-> then we have 343 for the number of X
            #-> then we have the number of tokens used for the encoding of the word
            #-> then ending it up with the <EOS> token
        #tokenized_prompts -> [100,77]  (for each class)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            #so somehow using the above tokenized embeddings, we just get these embeddings
            #embeddings -> #[100,77,512]   (77 token long, 512 for each token)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        #now that part of the embedding chosen is gonna be [100,512] -> more like the 1st token for each of the 100 classes, hence the SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        #So this is going to have the CLS, EOS and the remaining of the 0s

        '''TO EXPLAIN IN SHORT WE DID IN THE TOP HERE
        1) Make tokenized_prompts to have the tokenized form of ['X X X X X cls.'] -> [100,77]
        2) Make embedding out of that ->[100,77,512]
        3) Now from these embeddings, just take out what we think is the <SOS>, <CLS> and <EOS>
        4) Append these to the ctx (learnable context vectors) according to the rule
        '''

        self.n_cls = n_cls    #100
        self.n_ctx = n_ctx    #16
        self.tokenized_prompts = tokenized_prompts  #SO AGAIN THIS IS FOR ALL THE CLASS NAMES THAT WE HAVE [100,77]
        self.name_lens = name_lens   #LIST OF 100 ELEMENTS HAVING THE LENGTH OF EACH EMBEDDING
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION if not tem_init else "template"  #'END' IN OUR CASE

    def forward(self):
        ctx = self.ctx  
        #THE RANDOMLY INIT -> [16,512]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            #gonna be [100,16,512]

        prefix = self.token_prefix
        #prefix -> [100,1,512] -> the SOS embeddings for all the 100 classes
        suffix = self.token_suffix
        #suffix -> [100,60,512 ] -> the CLS+EOS embeddings for all the 100 classes

        #put them all together -> [100,77,512]

        if self.class_token_position == "end":   #this itself
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, remaining (60), dim)
                ],
                dim=1,
            )
        
        #so more like the whole point of doing this, is to not have the X X X X as the context, but 
        #rather these context vectors which are trainable. The prefix and the suffix are gonna be 
        #more or less the same
            
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        elif self.class_token_position == 'template':
            half_n_ctx = self.cls_loc
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        #prompts -> [100,77,512]  -> basically for each of the 100 classes, we have 77 token long which is gonna be 512 embedding long

        #so to put it in short, for each of the 100 classes, were going to be having the prompt such that its of shape [77,512], where 1st -> <SOS>
        # next 16 -> the learnable tokens  [PRETTY MUCH THE PROMPT LEARNER], and the remaining 60 -> <CLS> <EOS> and ending up with 0s
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  VPT DEEP!                   #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX                              #10
        self.dtype = clip_model.dtype        
        self.ctx_dim = clip_model.visual.conv1.out_channels             #768
        self.clip_imsize = clip_model.visual.input_resolution           #224
        self.cfg_imsize = cfg.INPUT.SIZE[0]                             #224
        self.layers = clip_model.visual.transformer.layers              #12
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 #11
        self.meta_net_num = self.layers - self.bottom_limit             #1
        
        vis_dim = clip_model.visual.output_dim              #512
        
        ctx_vectors = torch.empty(self.bottom_limit, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        #ctx_vectors - > [11,10,768]

        #so like in the original VPT it was [12,10,768] -> the other layer will be coming up later
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self, batch_size):
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1, -1) # batch layers n_ctx feature 
        #so if 32 was fed into the learner, we would be getting [32,11,10,768]
        return ctx

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj
        
    def forward(self,x):
        if self.proj is not None:
            x = x @ self.proj
        return x
    

#SO NOW HERE THE ATTENTION MECHANISM HAS BEEN CODED OUT ON ITS OWN RATHER THAN TRYING TO GET IT FROM THE CLIP.py FILE
class Attention(nn.Module):
    def __init__(self, clip_model, min=0.02):
        super().__init__()
        # self.bias = 
        self.min = min
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution  #224
        self.kmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype) #[768,32]
       
        self.qmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype) #[768,32]
        self.vmlp = nn.Linear(self.ctx_dim, self.ctx_dim,bias=False, dtype=clip_model.dtype)  #[768,768]
        
    def forward(self, q, k, v):
        #q -> [10,32,768]
        #k,v -> [60,32,768]
        
        q = q.permute(1,0,2); k=k.permute(1,0,2); v=v.permute(1,0,2)
        q = self.qmlp(q); k = self.kmlp(k)
        u = torch.bmm(q, k.transpose(1,2))
        #q-> [32,10,32] k->[32,60,32] u->[32,10,60]

        u = u / (math.sqrt(q.shape[-1]))
        
        attn_map = F.softmax(u, dim=-1)
        #attn_map ->[32,10,60]
        output = torch.bmm(attn_map, v)
        output = self.vmlp(output)
        #output -> [32,10,768]
        
        return output.permute(1,0,2), attn_map
    
class CAVPT(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_ctx = cfg.TRAINER.VPT.N_CTX                     #10
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels    #768
        self.clip_imsize = clip_model.visual.input_resolution  #224
        self.cfg_imsize = cfg.INPUT.SIZE[0]                    #224
        self.layers = clip_model.visual.transformer.layers     #12

        #10
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        
        #11
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
        
        #1
        self.meta_net_num = self.layers - self.bottom_limit
        
        vis_dim = clip_model.visual.output_dim           #512


        #basically going to be a module now having just one linear layer (cos met_net_num = 1)
        #each linear having -> [512,768]
        self.meta_nets = nn.ModuleList([nn.Linear(vis_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
        
        if cfg.TRAINER.COOP.PREC == 'fp16':
            for i in range(self.meta_net_num):
                self.meta_nets[i].half()
    
        
        '''IF WE'RE NOT CHANGING THE INITIAL CONFIGURATION OF THE PROGRAM, i WILL BE 1 ITSELF, AND WE'LL HAVE ONLY 1 ELEMENT IN ALL THESE MODULE LISTS, HENCE NO POINT OF IT AS SUCH'''

        #creating a module having one attention layer (NOTHING TO DO WITH CLIP, BUT JUST FOR THE SIZES)
        self.attns = nn.ModuleList([Attention(clip_model) for _ in range(self.layers-self.bottom_limit)])

        #can just say similarly for these 2 too
        #LayerNorm [768]
        self.lns = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])
        #Linear [768,100]
        self.classfiers = nn.ModuleList([nn.Linear(self.ctx_dim, len(classnames), bias=False) for _ in range(self.layers - self.bottom_limit)])
        # self.prompt_linear = nn.ModuleList([nn.Linear(self.ctx_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])

   
        ctx_vectors = torch.empty(self.layers - self.bottom_limit, 10, self.ctx_dim, dtype=self.dtype)
        '''NOTE THAT THESE VECTORS WERE MADE JUST FOR THE CAVPT PART, AND SINCE THE IMAGE HAS BEEN THROUGH 11 (DEAFULT) LAYERS ALREADY WITH THE VPT CONTEXT,
        THE CAVPT CONTEXT ONLY NEEDS TO HELP FOR ANOTHER 1 LAYER'''
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self, class_token, class_prompt, i):
        class_token = class_token.detach()
        '''CLASS TOKEN IS BASICALLY THE IMAGE FEATURES ITSELF AND CLASS PROMPT IS THE TEXT FEATURE'''

        #class token -> [50,32,768]  image features ready for the next layer cavpt layer of transformer
        #class prompt -> [32,10,512]  #best 10 text features

        # class_token = self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)
        
        class_prompt = self.meta_nets[i](class_prompt).permute(1, 0, 2)
        #class prompt -> [10,32,768]
        '''PASSING THE TEXT FEATURE THROUGH SOME LINEARS, TURNS OUT TO GET THE SAME SHAPE AS THE CLASS TOKEN'''
        
        '''TO THE IMAGE WE PUT IN THE ONLY CTX LAYER AND THEN CHANGE SHAPE A LITTLE'''
        class_token = torch.cat([class_token, self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)])
        #class token -> [60,32,768]

        x = class_prompt
        
        '''WHAT WE DO NOW IS THAT WE PASS THIS NEW IMAGE WITH CLASS CONTEXT, AND THE TEXT FEATURES IN THE ATTENTION LAYER, TO GET THE NEW IMAGE FEATURES'''
        class_prompt, attn_map = self.attns[i](class_prompt, class_token, class_token)
        #class_prompt -> [10,32,768]
        class_prompt4logits = self.lns[i](class_prompt)
        #class_prompt4logits -> [10,32,768]
        
        logits = self.classfiers[i](class_prompt4logits)
        class_prompt = self.lns2[i](class_prompt + x)

        #logits -> [10,32,100]  class_prompt-> [10,32,768]
        
        return class_prompt, logits, attn_map
        
    
class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX                  #10
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution #224
        self.cfg_imsize = cfg.INPUT.SIZE[0]                   #224
        self.layers = clip_model.visual.transformer.layers    #12

        # model
        transformer = clip_model.visual.transformer            
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers

        self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)

        ### THIS PART GOT ADDED IN, REMAINING WAS JUST SAME FROM VLP 
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        #10
        self.n_ctx = self.n_ctx
        #10
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
        #11
        
        self.extractor = CAVPT(cfg, classnames, clip_model).half()
        #JUST MADE A CAVPT OBJECT
        

    def forward(self, x, text_feature, need_attn=False):

        #x ->
        #text_feature ->

        ctx = self.ctx_learner(x.shape[1]) # batch layers n_ctx feature
        #ctx be [32,11,10,768]
        ctx = ctx.permute(1, 2, 0, 3)
        #ctx -> [11,10,32,768]

        # top_ctx = top_ctx.permute(1, 2, 0, 3)
        
        n_ctx = self.n_ctx   
        #10
        
        # ctx = bottom_ctx
        
        for i in range(self.bottom_limit):    #for loop goes on 11 times
            # print(ctx[i].shape, x.shape)
            '''SAME WORKING AS IN VPT -> PUT THE CTX VECTORS IN THE END, PASS THROUGH THE RESBLOCKS (ATTENTION)
            AND THEN FINALLY REMOVE THE LAST 10 FOR THE NEXT ROUND'''
            x = torch.cat([x, ctx[i]], dim=0)
            #[60,32,768]
            x = self.resblocks[i](x)
            x = x[:-n_ctx, :, :]
            # print("bottom", x.shape)
        
        #x->[50,32,768]
            
        n_ctx = self.class_prompt_num
        #10
        
        layer_logits = []
        
        for i in range(self.layers-self.bottom_limit):   #LOOP RUNS ONCE
            class_token = x
            # class_prompt = ctx[i][self.n_ctx:, :, :] # class_prompt_num, batch_size, feature.
            '''SO WHAT WE'RE DOING IS PASSING THE PRE-FINAL (11TH) LAYER X ALONG WITH THE TEXT FEATURE IN THE CAVPT PART  [i is gonna be 1 itself]'''
            class_prompt, layer_logit, attn_map = self.extractor(class_token, text_feature, i)
            

            layer_logits.append(layer_logit.unsqueeze(0))
            #gonna be a list of the layer_logits

            x = torch.cat([x, class_prompt], dim=0)
            '''now instead of putting in some ctx[i], we're putting in the class prompt which came from the CAVPT and then do same process'''
            if(need_attn):
                attn=self.resblocks[i+self.bottom_limit(x,True)]
                return attn

            else:
                x = self.resblocks[i+self.bottom_limit](x)
                if n_ctx != 0:
                    x = x[:-n_ctx, :, :]
            
        '''FINALLY GIVING BACK THE X AFTER HAVING PASSED IT THROUGH ALL THE LAYERS OF THE VISION ENCODER BUT LIKE THIS
        1) 11 OF THOSE LAYERS WERE DONE USING THE CONTEXT FROM THE CONTEXT LEARNER (ctx)
        2) LAST ONE WAS DONE USING THE CLASS PROMPT FROM THE CAVPT
        
        IN THIS CASE THE LAYER_LOGITS WILL JUST BE A SINGLE ELEMENT LONG, HAVING THE LAYER LOGITS OF THE LAST LAYER
        '''
        
        return x, layer_logits, attn_map


class ImageEncoder_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
        self.ln_post = clip_model.visual.ln_post
        # self.proj = clip_model.visual.proj
        self.proj = ProjLearner(clip_model)
        
    def forward(self, x, text_feature):
        '''SO THE WHOLE IMAGE ALONG WITH THE TOP 10 TEXT FEATURES COME IN NOW'''
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #x -> [32,768,7,7]
       
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        #x-> [32,49,768]

        # class_embedding is class token.
        '''adding the class token here'''
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #x -> [32,50,768]
        x = x + self.positional_embedding.to(x.dtype)
 
        x = self.ln_pre(x)
        #x-> [32,50,768]
        

        x = x.permute(1, 0, 2)  # NLD -> LND
        #x -> [50,32,768]
        x, layer_logits, attn_map= self.transformer(x, text_feature)
        #x->[50,32,768]
        #layer_logits -> list containing an element of [1,10,32,100]
        '''THE TEXT FEATURES [TOP-10] WILL ONLY BE USED BY THE CAVPT MECHANISM'''
        x = x.permute(1, 0, 2)  # LND -> NLD
        #x -> [32,50,768]

        '''SO THIS X CAME OUT FROM ALL THE LAYERS OF THE IMAGE ENCODER, SOMETIMES WENT IN WITH THE LEARNABLE CONTEXT, SOMETIMES WITH THE CAVPT CONTEXT'''
       
        '''AFTER ALL THESE LAYERS, WE JUST TAKE UP THE CLASS TOKEN FROM THE FEATURES, AND THEN PASS THAT ALONG THE PROJECTION LAYER'''
        x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.
        #x-> [32,768]
        
        x = self.proj(x)
        #x->[32,512]

        '''RETURING BACK X AND THE LIST OF LAYER_LOGITS FROM THE TRANSFORMER'''
        return x, layer_logits,attn_map


class CustomCLIP_Selected_CoVPTDeep(nn.Module):
    def __init__(self, cfg, classnames, clip_model, devices):
        super().__init__()
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        #10
        prompts = []
        # for temp in IMAGENET_TEMPLATES_SELECT:
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts += [temp.format(c.replace("_", " ")) for c in classnames]
        #so at this point, prompts is a 100 el long list of prompts like 'a photo of a {}' with the classname
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        #prompts -> [100,77], so where each of the 100 elements now get tokenized to 77 long list, where the first el will be SOS, and 
        #then the encoding for the prompt (majorly the tokens itself) -> followed by the cls and then the <EOS>, remaining gon be 0s
       
        clip_model.to(devices)
        prompts = prompts.to(devices)
        self.devices=devices
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            #now pass in these text tokens into clip text encoder -> gives out features for the text
            #text_features -> [100,512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        clip_model.to('cpu')
        self.text_features = nn.Parameter(text_features)
        self.classnames=classnames
        self.cfg=cfg
        self.clip_model=clip_model
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #this is gonna be the 77 long prompt for each class, but before embedding, so [100,77], but have the 'X X X ' instead of the learner vector

        # visual (going to be defining 2 image encoders (VPTD and clip -> mainly for ground test))
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
        self.zeroshot_clip_image_encoder = clip_model.visual
        # visual end (going to defining 1 itself)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        '''So the first part in this forward is to get to know the top-10 text features which are in line with the image'''
        '''and those indices be stored in indices'''
        #image -> [32,3,224,224]
        image = image.to(next(self.image_encoder.parameters()).device)
        #what this does is sets the image to either cpu or gpu depending on what the image encoder was set to
        #size gon remain the same [32,3,224,224]
        with torch.no_grad():
            zeroshotclip_image_feature = self.zeroshot_clip_image_encoder(image.type(self.dtype))
            zeroshotclip_image_feature = zeroshotclip_image_feature / zeroshotclip_image_feature.norm(dim=-1, keepdim=True)
            #zsclip -> [32,512]
            
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * zeroshotclip_image_feature @ self.text_features.t()
            #now then multiplying it with the text features, we get the logits
            #logits -> [32,100]
            _, indices = torch.sort(logits, descending=True)
            #so now indices is going to have the indices of the logits sorted in descending
            
            indices = indices[:, :self.class_prompt_num]
            #and now we choose only the top 10 of this for every image in the batch
            
            # mask = indices==label.unsqueeze(0).expand()
            selected_text_features = self.text_features[indices]
        
        condition=False

        if(image.shape[0]==1):
            if self.cfg.EVAL.HAND_CRAFT:
                condition=True
            else:
                condition=False
        elif(self.cfg.TRAIN.HAND_CRAFT):
            condition=True
        
     
        # SO NOW USING THIS CODE EVEN FOR THE EVALUATION (IN CASE OF THE CLASSIFICATION FUNCITON, SO ONE SMALL UPDATE IN LOGIC)
        if(condition):     #SET TRUE FOR H1, SET FALSE FOR L1
          text_features=self.text_features
          
        else:
          prompts = self.prompt_learner()
          tokenized_prompts = self.tokenized_prompts
          text_features = self.text_encoder(prompts, tokenized_prompts)
        
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        #[100,512]

        '''NOW TO GET THE IMAGE FEATURES FROM THE IMAGE ENCODER (VPTD) TO WHICH WE'LL PASS IN 2 THINGS
        1) IMAGE [32,3,224,224]
        2) TOP 10 TEXT FEATURES [10,512]
        '''
        image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features[indices])
        

        #ONCE WE HAVE THE FINAL IMAGE FEATURES, WE JUST MULTIPLY IT WITH THE TEXT FEATURES (FULL 100, NOT TOP 10)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        #THIS ONE IS FOR VPT+COOP
        #print('Image features', image_features.shape)
        #print('Text features', text_features_norm.shape)
        logits = logit_scale * image_features @ text_features_norm.t()
        
        #THIS IS ONLY FOR COOP
        logits1 = logit_scale * zeroshotclip_image_feature @ text_features_norm.t()
        # loss1 = F.mse_loss(text_features, self.text_features)
        #THIS IS ONLY FOR VPT
        logits2 = logit_scale * image_features @ self.text_features.t()

        #logits, logits1, logits2 -> [32,100]
        
        #print(logits.shape,'LOGITS SHAPE')

        return logits, layer_logits, indices, logits1, logits2 , attn_map
    

    '''GOAL OF THIS FUNCTION IS TO USE IT FOR THE TEST PART ONLY (WHERE WE'D LIKE TO PASS IN THE LABEL OF THE CLASS TOO) -> NICELY DONE
    USING THE label=None, AND ALSO WELL APPROACHED IN THE get_heatmap FUNCTION'''
    def inference(self, image, label=None):
        '''So the first part in this forward is to get to know the top-10 text features which are in line with the image'''
        '''and those indices be stored in indices'''
        #image -> [32,3,224,224]
        image = image.to(next(self.image_encoder.parameters()).device)      
        
        '''HAND CRAFTER PROMPT INFERNENCE'''
        
        if(self.cfg.EVAL.HAND_CRAFT==True):    #SET TRUE FOR H2, SET FALSE FOR L2
        
          temp=CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
          prompts=[temp.format(label)]
          prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.devices)
          
          #print(prompts.shape)
          text_features = self.clip_model.encode_text(prompts)
          
          text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
  
          #print('text features',text_features.shape)
          
          image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features.unsqueeze(0))
        
        else:
          '''TRAINED PROMPT INFERENCE'''
          #print('INFERENCE USING RANDOM PROMPTS')          
          prompts = self.prompt_learner()
          #[2,77,512]
          tokenized_prompts = self.tokenized_prompts
          #[2,77]
          text_features = self.text_encoder(prompts, tokenized_prompts)
          #[2,512]
          
          text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        
          #SO BASICALLY PASSING IN ONLY THE TEXT FEATURE OF THE TUMOR TEXT, BUT NEED TO BE A 3D VECTOR, SO UNSQUEEZING AS WELL
          #image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features[1].unsqueeze(0).unsqueeze(0))
          image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features.unsqueeze(0))
        
        
        '''COMMON PART STARTS AFTER THIS'''
        
        #image_features->[32,512]
        #layer_logits -> list having one element of [1,10,32,100]

        #ONCE WE HAVE THE FINAL IMAGE FEATURES, WE JUST MULTIPLY IT WITH THE TEXT FEATURES (FULL 100, NOT TOP 10)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        
        logit_scale = self.logit_scale.exp()
        
        #print('Image features', image_features.shape)
        #print('Text features', text_features_norm.shape)
        logits = logit_scale * image_features @ text_features_norm.t()
        #print('INFERENCE LOGITS',logits.shape)

        #print(logits.shape,'LOGITS SHAPE')
        return logits
    
# end




@TRAINER_REGISTRY.register()
class DPT(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    
    def model_inference(self, input):
        return self.model.forward(input)[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        self.pretrain_c = cfg.PRETRAIN.C
        self.alpha = cfg.TRAINER.ALPHA

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # ================================== #
        #              VPT DEEP              #
        # ================================== #
        print("Building custom CLIP VPT Deep")
        self.model = CustomCLIP_Selected_CoVPTDeep(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        
        '''
        for name, param in self.model.named_parameters():
            if "image_encoder.transformer.ctx_learner" not in name and 'extractor' not in name and "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
        '''
        
        

        self.model.to(self.device)
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        if cfg.DATASET.NAME=='ImageNet':
            opt_cfg.WARMUP_EPOCH=1
        opt_cfg.freeze()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.image_encoder.transformer.ctx_learner, opt_cfg)
        self.sched = build_lr_scheduler(self.optim, opt_cfg)
        self.optim1 = build_optimizer(self.model.image_encoder.transformer.extractor, opt_cfg)
        self.sched1 = build_lr_scheduler(self.optim1, opt_cfg)
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        # opt_cfg.WARMUP_TYPE="constant"
        opt_cfg.WARMUP_EPOCH=1
        opt_cfg.LR = 0.002
        opt_cfg.freeze()
        self.optim2 = build_optimizer(self.model.prompt_learner, opt_cfg)
        self.sched2 = build_lr_scheduler(self.optim2, opt_cfg)
        self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)
        self.register_model("image_encoder.transformer.extractor", self.model.image_encoder.transformer.extractor, self.optim1, self.sched1)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim2, self.sched2)

  

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        #print('img shape ',label.shape)
        #image -> [32,3,224,224]
        #label -> [32]
        prec = self.cfg.TRAINER.COOP.PREC

        #NOT GONNA BE OUR CASE
        if prec == "amp":
            with autocast():
                output = self.model.forward(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
       
        #OUR DEAL RIGHT HERE LAD
        else:
            
            real_label = label
            
            
            output=self.model.forward(image)[0]
            
            layer_logits=self.model.forward(image)[1]
            indices=self.model.forward(image)[2]
            output1=self.model.forward(image)[3]
            output2=self.model.forward(image)[4]
            
        

            '''LABEL IS BASICALLY GOING TO BE THE TOKEN OF THE RIGHT CLASS, AND WE CAN USE THE F.cross_entropy TO FIND THE ENTROPY LOSS IN THIS CASE'''
            if self.epoch < self.pretrain_c: # output1=text, output2=image
                loss = F.cross_entropy(output1, label) + F.cross_entropy(output2, label) + 0.1*F.cross_entropy(output, label)
            else:
                loss = F.cross_entropy(output, label)

            '''THIS SECTION ONWARDS, THEY ARE ALSO ACCOUNTING FOR ANOTHER LOSS FUNCTION, WHICH GETS ADDED TO THE LOSS FUNCTION ABOVE
            NOT REALLY REQUIRED AT THIS POINT OF TIME, SO NOT GETTING DEEPER INTO THIS DRILL'''
            
            layers = len(layer_logits)
            layer_logits = torch.cat(layer_logits, dim=0).permute(2, 0, 1, 3).reshape([-1, len(self.classnames)]) # batch, layer, class_prompt_num, class_num
            batch_target = torch.tensor([1/self.num_classes] * len(self.classnames), dtype=torch.float16).unsqueeze(0).expand(layer_logits.shape[0], -1).to(self.device)
            
            
            label = label.reshape([-1, 1]).expand(-1, self.class_prompt_num)
            tmp = label == indices 
            tmp = tmp.unsqueeze(1).expand(-1, layers, -1).reshape([-1])
            label = label.unsqueeze(1).expand(-1, layers, -1)
            one_hot_code = F.one_hot(label.reshape([-1]), len(self.classnames))
            tmp = tmp.unsqueeze(1).expand(-1, len(self.classnames))
            one_hot_code[tmp==False] = 0
            
            batch_target[tmp] = 0
            batch_target = batch_target+one_hot_code
            
            layer_logits = layer_logits[tmp]
            batch_target = one_hot_code[tmp].to(torch.float16)
            
            if self.class_prompt_num != 0 and layer_logits.shape != torch.Size([0]):
                loss = loss + self.alpha * F.cross_entropy(layer_logits.reshape([-1, self.num_classes]), batch_target.reshape([-1, self.num_classes]))
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "tmp": tmp.item(),
            "acc": compute_accuracy(output, real_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            
            if 'classfiers.0.weight' in state_dict:
                for i in range(12 - self.cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT+1):
                    del state_dict[f"classfiers.{i}.weight"]
                # del state_dict["classfiers.0.bias"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
    
    def create_maps(self,path):

        if self.cfg.EVAL.CLASSIFY:
            print("CREATING MAPS THROUGH CLASSIFICATION APPROACH")
        
        batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE

        total_length=len(self.dm.dataset.test)       
        print('TOTAL TEST IMAGES',total_length)
        num_batches=total_length//batch_size
        extra=total_length%batch_size
        print('TOTAL BATCHES',num_batches)     
        
        mask_path=os.path.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME, "DDPT_MASK")
    # Construct the base path
        file_path_healthy=os.path.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME, "DDPT_Classified_Healthy.txt")
        file_path_unhealthy=os.path.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME, "DDPT_Classified_Unhealthy.txt")


        # Ensure parent directories exist
        create_directory(mask_path)
              
        batch_num=0

        healthy_store={}
        unhealthy_store={}
                
        print('CREATING MAPS NOW')
        progress_bar = tqdm(total=num_batches+1)

        
        for batch in self.test_loader:
        
            if(batch_num==0):
              for img_num in range(batch_size):
                  
                img=batch['img'][img_num]
                #label=batch['label'][img_num]
                impath=batch['impath'][img_num]
                
                if self.cfg.EVAL.CLASSIFY:
                    #This is just calculated to see for the classification
                    out = self.model(torch.unsqueeze(img, dim=0))
                    out=out[0][0]
                    prob = F.softmax(out, dim=0)
                    if prob[1] > prob[0]:     # Healthy Case
                        #healthy.append(impath)
                        healthy_store[impath]=prob[1]
                    else:
                        unhealthy_store[impath]=prob[0]
                    _=self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')  #Re-running again so we are sure that only the Tumour class prompt is sent in
                    save_mask(img,impath,mask_path)

                      
       
            else:
              for img_num in range(extra):                    
                img=batch['img'][img_num]
                #label=batch['label'][img_num]
                impath=batch['impath'][img_num]

                
                if self.cfg.EVAL.CLASSIFY:
                    #This is just calculated to see for the classification
                    out = self.model(torch.unsqueeze(img, dim=0))
                    out=out[0][0]
                    prob = F.softmax(out, dim=0)
                    if prob[1] > prob[0]:     # Healthy Case
                        #healthy.append(impath)
                        healthy_store[impath]=prob[1]
                    else:
                        unhealthy_store[impath]=prob[0]
                    _=self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')  #Re-running again so we are sure that only the Tumour class prompt is sent in
                    save_mask(img,impath,mask_path)
                
            
            batch_num+=1
            progress_bar.update(1)
        
        with open(file_path_healthy, "w") as file:
            for key, value in healthy_store.items():
                file.write(f"{key}: {value}\n")  

        with open(file_path_unhealthy, "w") as file:
            for key, value in unhealthy_store.items():
                file.write(f"{key}: {value}\n")  


    def create_maps_eval(self,path):

        if self.cfg.EVAL.CLASSIFY:
            print("CREATING MAPS THROUGH CLASSIFICATION APPROACH")
        
        batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE

        total_length=len(self.dm.dataset.test)       
        print('TOTAL TEST IMAGES',total_length)
        num_batches=total_length//batch_size
        extra=total_length%batch_size
        print('TOTAL BATCHES',num_batches)     

        folder_paths={}
        
        folder_paths['images']=  f'inference/{path}/img'     #2
        folder_paths['prediction']=f'inference/{path}/pred'  #1
        folder_paths['masks']=f'inference/{path}/mask'       #3


        data, shots, cat = path.split('/')

        
    # Construct the base path
        base_path = os.path.join('inference', data, shots, cat)

        # Ensure parent directories exist
        create_directory(os.path.join('inference', data))
        create_directory(os.path.join('inference', data, shots))
        create_directory(base_path)

        # Set up and return folder paths
        setup_folders(base_path)
        
        batch_num=0
        
                
        print('CREATING MAPS NOW')
        progress_bar = tqdm(total=num_batches+1)

        healthy=[]
        unhealthy=[]
        
        for batch in self.test_loader:
        
            if(batch_num!=num_batches):
              for img_num in range(batch_size):
                  
                img=batch['img'][img_num]
                label=batch['label'][img_num]
                impath=batch['impath'][img_num]
                
                if self.cfg.EVAL.CLASSIFY:
                    #This is just calculated to see for the classification
                    out = self.model(torch.unsqueeze(img, dim=0))
                    logits=out[0]
                    if logits[0][1] > logits[0][0]:     # Healthy Case
                        #healthy.append(impath)

                        logits=self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')  #Re-running again so we are sure that only the Tumour class prompt is sent in
                        if label==1:
                            process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=False) 
                        else:
                            process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=True) 

                elif label == 1:
                   logits = self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')
                   process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=False)

                
                  
            
            else:
              for img_num in range(extra):                    
                img=batch['img'][img_num]
                label=batch['label'][img_num]
                impath=batch['impath'][img_num]

                logits=self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')
                process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=False) 

                '''
                if self.cfg.EVAL.CLASSIFY:
                    #this is just calculated to see for the classification
                    out = self.model(torch.unsqueeze(img, dim=0))
                    logits=out[0]
                    if logits[0][1] > logits[0][0]:
                        healthy.append(impath)
                        logits=self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')  #Re-running again so we are sure that only the Tumour class prompt is sent in
                        if label==1:
                            process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=False) 
                        else:
                            process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=True) 
                    # REMOVE THESE 2 LINES
                    else:
                        unhealthy.append(impath)

                elif label == 1:
                   logits = self.model.inference(torch.unsqueeze(img, dim=0), 'Tumour')
                   process_image(img, impath, folder_paths,dataset=self.cfg.DATASET.NAME,input_size=self.cfg.INPUT.SIZE,mis_pred=False)
                '''

            '''
            data = {"healthy": healthy, "unhealthy": unhealthy}

            with open("lists.json", "w") as f:
                json.dump(data, f)    
            '''

            batch_num+=1
            progress_bar.update(1)
            #print(f'BATCHES DONE {batch_num}/{num_batches}')

        print('HEALTHY IMAGES', len(healthy))
        print('UNHEALTHY IMAGES', len(unhealthy))
        
              
    def generate_metrics(self, folder, threshold, median):
        """
        Generate evaluation metrics for the test dataset.
        """

        print(f"Setting threshold for ground masks as {threshold}")

        # Initialize aggregated metrics
        aggregated_metrics = {
            'dice': {'0.4': 0, '0.5': 0, '0.6': 0},
            'iou': {'0.4': 0, '0.5': 0, '0.6': 0},
            'auprc': {'0.4': 0, '0.5': 0, '0.6': 0},
            'f1': {'0.4': 0, '0.5': 0, '0.6': 0},
            'recall': {'0.4': 0, '0.5': 0, '0.6': 0},
            'precision': {'0.4': 0, '0.5': 0, '0.6': 0},
        }


        image_file_names=get_file_names(folder)

        progress_bar = tqdm(total=len(image_file_names))
        threshold_remove=0

        for file_name in image_file_names:
            metrics = eval_metrics(file_name, folder, threshold, median)
            if metrics==False:
                threshold_remove+=1
                progress_bar.update(1)
                continue
            for metric_type in aggregated_metrics:
                for threshold_key in aggregated_metrics[metric_type]:
                    aggregated_metrics[metric_type][threshold_key] += metrics[metric_type][threshold_key]
            #print(aggregated_metrics)
            progress_bar.update(1)

        progress_bar.close()
        print(f"REMOVED {threshold_remove} IMAGES DUE TO THRESHOLDING")


        # Save results
        folder_name = folder.replace("/", "-")
        save_path = f"result-{folder_name}-{threshold}{'-median' if median else ''}.txt"
        write_results_to_file(save_path, len(image_file_names)-threshold_remove,aggregated_metrics)

        print(f"DONE SAVING: {save_path}")
        
    
      
         
    

# from audioop import tomono
# import copy
# import math
# from os import TMP_MAX
# import os.path as osp
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# from metrics.metrics import eval_metrics

# from torchvision import transforms
# from PIL import Image

# import numpy as np

# import shutil
# import os

# import cv2

# import json

# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

# _tokenizer = _Tokenizer()

# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",

#     #Brain dataset that I have just added
#     "Brain101": "a photo of a {}.",
#     "Brain201": "a photo of a {} brain.",
    
#     "Brain301": "a photo of a {} brain.",
#     "Brain302": "a photo of a {} brain.",
#     "Brain303": "a photo of a {} brain.",
#     "Brain304": "a photo of a {} brain.",
    
#     "Brain401": "a photo of a {} brain.",
#     "Brain402": "a photo of a {} brain.",
#     "Brain403": "a photo of a {} brain.",
    
#     "Brain501": "a photo of a {} brain.",
#     "Brain601": "a photo of a {} brain.",
#     "Liver101": "a photo of a {}.",
#     "BraTS23": "a photo of a {} Brain.",
#     "ATLAS": "a photo of a {}.",
# }

# # ================================================================= #
# # 这个是VPT DEEP+CoOp（Frozen）+CLIP（Frozen）的实现。                #
# # ================================================================= #


# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())

#     return model

# # ++++++++++++++++++++++++++++++++++++++++++++ #
# #                  Pure CoOp!                  #
# # ++++++++++++++++++++++++++++++++++++++++++++ #
# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts):
        
#         #prompts->[100,77,512] (these are what have the learnable ctx vectors)
#         #tokenized_prompts ->[100,77] -> now these are the prompts before learners and have XXXX
       
#         x = prompts + self.positional_embedding.type(self.dtype)
#         #x -> [100,77,512]
       

#         x=x.permute(1,0,2)
#         x=self.transformer(x)
       
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         #x->[100,77,512]

#         # x.shape = [batch_size, n_ctx, transformer.width] hmmmm what??

#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         #so in this step kinda like were taking only a few features after doing the linear layer, and then passing through the projection layer

#         #x-> [100,512]

#         '''IN SHORT WHAT HAPPENS IN THIS TEXT ENCODER IS THAT THE PROMPTS (LEARNABLE) ARE PASSED THROUGH THE CLIP TEXT TRANSFORMER, AND WE CHOOSE
#         ONLY A PART OF THE OUTPUT, BASED ON THE TOKENIZED PROMPTS (TOKENS OF 'X X X X X CLS.') , WHICH WE THEN PASS THROUGH THE PROJECTION LAYER'''
#         return x


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, label=None, devices=None):
#         super().__init__()
#         if(label):
#             n_cls=1
#             classnames=[label]
#             print('PROMPT LEARNER', classnames)
#         else:
#             n_cls = len(classnames)                             #p self explanatory
#         n_ctx = cfg.TRAINER.COOP.N_CTX                      #'M' in the paper
#         ctx_init = cfg.TRAINER.COOP.CTX_INIT                # Going to be ' ' in our case
#         # tem_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
#         tem_init = False
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]       #well see soon what this does
#         clip_imsize = clip_model.visual.input_resolution    #image input (224)
#         cfg_imsize = cfg.INPUT.SIZE[0]                      #224
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         #[NOT GOING TO BE OUR CASE]
#         if ctx_init:
#             # use given words to initialize context vectors 
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#             prompt_prefix = ctx_init

#         #NOPE AGAIN
#         elif tem_init:     
#             target_nctx = n_ctx
#             ctx_init = tem_init[:-1]
#             self.class_token_position = "template"
#             ctx_init = ctx_init.replace("_", " ")
#             ctx_init = ctx_init.split(' ')
#             if "{}" in ctx_init:
#                 self.cls_loc= ctx_init.index("{}")
#                 ctx_init.remove("{}")
#             elif "{}," in ctx_init:
#                 self.cls_loc= ctx_init.index("{},")
#                 ctx_init.remove("{},")
#             elif "{}." in ctx_init:
#                 self.cls_loc= ctx_init.index("{}.")
#                 ctx_init.remove("{}.")
#             n_ctx = len(ctx_init)
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             tmp = torch.empty(target_nctx-n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(tmp, std=0.02)
#             ctx_vectors = torch.cat([embedding[0, 1 : 1 + n_ctx, :], tmp], dim=0)
#             # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#             prompt_prefix = " ".join(ctx_init)+" "+" ".join(["X"]*(target_nctx-n_ctx))
#             n_ctx = target_nctx

#         # random initialization  [THHIS OVER HERE IS THE REAL DEAL SIR]
#         else:
           

#             #NOW WE WOULD DO THIS IF WE CLASS SPECIFIC CONTEXT, WHICH IS NOT THE CASE FOR US, SO WE MOVE AHEAD]
#             if cfg.TRAINER.COOP.CSC:            
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             #HERE IS FOR THE UNIFIED CONTEXT
#             else:
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#                 #ctx vectors -> [16,512]
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#             #prompt_prefix -> only gonna be 16 X followed by _ in the string

#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
#         #so this part has nothing to do with the classnames we set till now, its completely randomly initialized

#         classnames = [name.replace("_", " ") for name in classnames]
#         #we have 100 classnames like face, leopard etc
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         #not used here, but just a list of the length of the encodings of each name
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         #so basically its like a big list of items, wheres these items are basically those 16 Xs followed by the class and then '.'
#         #something like ['X X X X X X X X X X X X X X X X face.'] and 100 elements like this
        

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(devices)
#         #so basically just tokenizes the above prompts by doing this
#             #-> adds a <SOS> token first
#             #-> then we have 343 for the number of X
#             #-> then we have the number of tokens used for the encoding of the word
#             #-> then ending it up with the <EOS> token
#         #tokenized_prompts -> [100,77]  (for each class)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#             #so somehow using the above tokenized embeddings, we just get these embeddings
#             #embeddings -> #[100,77,512]   (77 token long, 512 for each token)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         #now that part of the embedding chosen is gonna be [100,512] -> more like the 1st token for each of the 100 classes, hence the SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
#         #So this is going to have the CLS, EOS and the remaining of the 0s

#         '''TO EXPLAIN IN SHORT WE DID IN THE TOP HERE
#         1) Make tokenized_prompts to have the tokenized form of ['X X X X X cls.'] -> [100,77]
#         2) Make embedding out of that ->[100,77,512]
#         3) Now from these embeddings, just take out what we think is the <SOS>, <CLS> and <EOS>
#         4) Append these to the ctx (learnable context vectors) according to the rule
#         '''

#         self.n_cls = n_cls    #100
#         self.n_ctx = n_ctx    #16
#         self.tokenized_prompts = tokenized_prompts  #SO AGAIN THIS IS FOR ALL THE CLASS NAMES THAT WE HAVE [100,77]
#         self.name_lens = name_lens   #LIST OF 100 ELEMENTS HAVING THE LENGTH OF EACH EMBEDDING
#         self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION if not tem_init else "template"  #'END' IN OUR CASE

#     def forward(self):
#         ctx = self.ctx  
#         #THE RANDOMLY INIT -> [16,512]
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#             #gonna be [100,16,512]

#         prefix = self.token_prefix
#         #prefix -> [100,1,512] -> the SOS embeddings for all the 100 classes
#         suffix = self.token_suffix
#         #suffix -> [100,60,512 ] -> the CLS+EOS embeddings for all the 100 classes

#         #put them all together -> [100,77,512]

#         if self.class_token_position == "end":   #this itself
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,     # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, remaining (60), dim)
#                 ],
#                 dim=1,
#             )
        
#         #so more like the whole point of doing this, is to not have the X X X X as the context, but 
#         #rather these context vectors which are trainable. The prefix and the suffix are gonna be 
#         #more or less the same
            
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i = ctx[i : i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,   # (1, name_len, dim)
#                         ctx_i,     # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
        
#         elif self.class_token_position == 'template':
#             half_n_ctx = self.cls_loc
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         else:
#             raise ValueError

#         #prompts -> [100,77,512]  -> basically for each of the 100 classes, we have 77 token long which is gonna be 512 embedding long

#         #so to put it in short, for each of the 100 classes, were going to be having the prompt such that its of shape [77,512], where 1st -> <SOS>
#         # next 16 -> the learnable tokens  [PRETTY MUCH THE PROMPT LEARNER], and the remaining 60 -> <CLS> <EOS> and ending up with 0s
#         return prompts


# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))

#         prompts = self.prompt_learner()
#         tokenized_prompts = self.tokenized_prompts
#         text_features = self.text_encoder(prompts, tokenized_prompts)

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()

#         return logits


# # ++++++++++++++++++++++++++++++++++++++++++++ #
# #                  VPT DEEP!                   #
# # ++++++++++++++++++++++++++++++++++++++++++++ #
# class VPTDeepPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         # hyper param
#         self.n_ctx = cfg.TRAINER.VPT.N_CTX                              #10
#         self.dtype = clip_model.dtype        
#         self.ctx_dim = clip_model.visual.conv1.out_channels             #768
#         self.clip_imsize = clip_model.visual.input_resolution           #224
#         self.cfg_imsize = cfg.INPUT.SIZE[0]                             #224
#         self.layers = clip_model.visual.transformer.layers              #12
#         self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 #11
#         self.meta_net_num = self.layers - self.bottom_limit             #1
        
#         vis_dim = clip_model.visual.output_dim              #512
        
#         ctx_vectors = torch.empty(self.bottom_limit, self.n_ctx, self.ctx_dim, dtype=self.dtype)
#         #ctx_vectors - > [11,10,768]

#         #so like in the original VPT it was [12,10,768] -> the other layer will be coming up later
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
        
#     def forward(self, batch_size):
#         ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1, -1) # batch layers n_ctx feature 
#         #so if 32 was fed into the learner, we would be getting [32,11,10,768]
#         return ctx

# class ProjLearner(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.proj = clip_model.visual.proj
        
#     def forward(self,x):
#         if self.proj is not None:
#             x = x @ self.proj
#         return x
    

# #SO NOW HERE THE ATTENTION MECHANISM HAS BEEN CODED OUT ON ITS OWN RATHER THAN TRYING TO GET IT FROM THE CLIP.py FILE
# class Attention(nn.Module):
#     def __init__(self, clip_model, min=0.02):
#         super().__init__()
#         # self.bias = 
#         self.min = min
#         self.dtype = clip_model.dtype
#         self.ctx_dim = clip_model.visual.conv1.out_channels # 768
#         self.clip_imsize = clip_model.visual.input_resolution  #224
#         self.kmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype) #[768,32]
       
#         self.qmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype) #[768,32]
#         self.vmlp = nn.Linear(self.ctx_dim, self.ctx_dim,bias=False, dtype=clip_model.dtype)  #[768,768]
        
#     def forward(self, q, k, v):
#         #q -> [10,32,768]
#         #k,v -> [60,32,768]
        
#         q = q.permute(1,0,2); k=k.permute(1,0,2); v=v.permute(1,0,2)
#         q = self.qmlp(q); k = self.kmlp(k)
#         u = torch.bmm(q, k.transpose(1,2))
#         #q-> [32,10,32] k->[32,60,32] u->[32,10,60]

#         u = u / (math.sqrt(q.shape[-1]))
        
#         attn_map = F.softmax(u, dim=-1)
#         #attn_map ->[32,10,60]
#         output = torch.bmm(attn_map, v)
#         output = self.vmlp(output)
#         #output -> [32,10,768]
        
#         return output.permute(1,0,2), attn_map
    
# class CAVPT(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.n_ctx = cfg.TRAINER.VPT.N_CTX                     #10
#         self.dtype = clip_model.dtype
#         self.ctx_dim = clip_model.visual.conv1.out_channels    #768
#         self.clip_imsize = clip_model.visual.input_resolution  #224
#         self.cfg_imsize = cfg.INPUT.SIZE[0]                    #224
#         self.layers = clip_model.visual.transformer.layers     #12

#         #10
#         self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        
#         #11
#         self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
        
#         #1
#         self.meta_net_num = self.layers - self.bottom_limit
        
#         vis_dim = clip_model.visual.output_dim           #512


#         #basically going to be a module now having just one linear layer (cos met_net_num = 1)
#         #each linear having -> [512,768]
#         self.meta_nets = nn.ModuleList([nn.Linear(vis_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
        
#         if cfg.TRAINER.COOP.PREC == 'fp16':
#             for i in range(self.meta_net_num):
#                 self.meta_nets[i].half()
    
        
#         '''IF WE'RE NOT CHANGING THE INITIAL CONFIGURATION OF THE PROGRAM, i WILL BE 1 ITSELF, AND WE'LL HAVE ONLY 1 ELEMENT IN ALL THESE MODULE LISTS, HENCE NO POINT OF IT AS SUCH'''

#         #creating a module having one attention layer (NOTHING TO DO WITH CLIP, BUT JUST FOR THE SIZES)
#         self.attns = nn.ModuleList([Attention(clip_model) for _ in range(self.layers-self.bottom_limit)])

#         #can just say similarly for these 2 too
#         #LayerNorm [768]
#         self.lns = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])
#         #Linear [768,100]
#         self.classfiers = nn.ModuleList([nn.Linear(self.ctx_dim, len(classnames), bias=False) for _ in range(self.layers - self.bottom_limit)])
#         # self.prompt_linear = nn.ModuleList([nn.Linear(self.ctx_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
#         self.lns2 = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])

   
#         ctx_vectors = torch.empty(self.layers - self.bottom_limit, 10, self.ctx_dim, dtype=self.dtype)
#         '''NOTE THAT THESE VECTORS WERE MADE JUST FOR THE CAVPT PART, AND SINCE THE IMAGE HAS BEEN THROUGH 11 (DEAFULT) LAYERS ALREADY WITH THE VPT CONTEXT,
#         THE CAVPT CONTEXT ONLY NEEDS TO HELP FOR ANOTHER 1 LAYER'''
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
        
#     def forward(self, class_token, class_prompt, i):
#         class_token = class_token.detach()
#         '''CLASS TOKEN IS BASICALLY THE IMAGE FEATURES ITSELF AND CLASS PROMPT IS THE TEXT FEATURE'''

#         #class token -> [50,32,768]  image features ready for the next layer cavpt layer of transformer
#         #class prompt -> [32,10,512]  #best 10 text features

#         # class_token = self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)
        
#         class_prompt = self.meta_nets[i](class_prompt).permute(1, 0, 2)
#         #class prompt -> [10,32,768]
#         '''PASSING THE TEXT FEATURE THROUGH SOME LINEARS, TURNS OUT TO GET THE SAME SHAPE AS THE CLASS TOKEN'''
        
#         '''TO THE IMAGE WE PUT IN THE ONLY CTX LAYER AND THEN CHANGE SHAPE A LITTLE'''
#         class_token = torch.cat([class_token, self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)])
#         #class token -> [60,32,768]

#         x = class_prompt
        
#         '''WHAT WE DO NOW IS THAT WE PASS THIS NEW IMAGE WITH CLASS CONTEXT, AND THE TEXT FEATURES IN THE ATTENTION LAYER, TO GET THE NEW IMAGE FEATURES'''
#         class_prompt, attn_map = self.attns[i](class_prompt, class_token, class_token)
#         #class_prompt -> [10,32,768]
#         class_prompt4logits = self.lns[i](class_prompt)
#         #class_prompt4logits -> [10,32,768]
        
#         logits = self.classfiers[i](class_prompt4logits)
#         class_prompt = self.lns2[i](class_prompt + x)

#         #logits -> [10,32,100]  class_prompt-> [10,32,768]
        
#         return class_prompt, logits, attn_map
        
    
# class Transformer_VPTD(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         # hyper param
#         self.n_ctx = cfg.TRAINER.VPT.N_CTX                  #10
#         self.dtype = clip_model.dtype
#         self.ctx_dim = clip_model.visual.conv1.out_channels # 768
#         self.clip_imsize = clip_model.visual.input_resolution #224
#         self.cfg_imsize = cfg.INPUT.SIZE[0]                   #224
#         self.layers = clip_model.visual.transformer.layers    #12

#         # model
#         transformer = clip_model.visual.transformer            
#         self.resblocks: nn.Sequential = transformer.resblocks
#         self.layers = transformer.layers

#         self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)

#         ### THIS PART GOT ADDED IN, REMAINING WAS JUST SAME FROM VLP 
#         self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
#         #10
#         self.n_ctx = self.n_ctx
#         #10
#         self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
#         #11
        
#         self.extractor = CAVPT(cfg, classnames, clip_model).half()
#         #JUST MADE A CAVPT OBJECT
        

#     def forward(self, x, text_feature, need_attn=False):

#         #x ->
#         #text_feature ->

#         ctx = self.ctx_learner(x.shape[1]) # batch layers n_ctx feature
#         #ctx be [32,11,10,768]
#         ctx = ctx.permute(1, 2, 0, 3)
#         #ctx -> [11,10,32,768]

#         # top_ctx = top_ctx.permute(1, 2, 0, 3)
        
#         n_ctx = self.n_ctx   
#         #10
        
#         # ctx = bottom_ctx
        
#         for i in range(self.bottom_limit):    #for loop goes on 11 times
#             # print(ctx[i].shape, x.shape)
#             '''SAME WORKING AS IN VPT -> PUT THE CTX VECTORS IN THE END, PASS THROUGH THE RESBLOCKS (ATTENTION)
#             AND THEN FINALLY REMOVE THE LAST 10 FOR THE NEXT ROUND'''
#             x = torch.cat([x, ctx[i]], dim=0)
#             #[60,32,768]
#             x = self.resblocks[i](x)
#             x = x[:-n_ctx, :, :]
#             # print("bottom", x.shape)
        
#         #x->[50,32,768]
            
#         n_ctx = self.class_prompt_num
#         #10
        
#         layer_logits = []
        
#         for i in range(self.layers-self.bottom_limit):   #LOOP RUNS ONCE
#             class_token = x
#             # class_prompt = ctx[i][self.n_ctx:, :, :] # class_prompt_num, batch_size, feature.
#             '''SO WHAT WE'RE DOING IS PASSING THE PRE-FINAL (11TH) LAYER X ALONG WITH THE TEXT FEATURE IN THE CAVPT PART  [i is gonna be 1 itself]'''
#             class_prompt, layer_logit, attn_map = self.extractor(class_token, text_feature, i)
            

#             layer_logits.append(layer_logit.unsqueeze(0))
#             #gonna be a list of the layer_logits

#             x = torch.cat([x, class_prompt], dim=0)
#             '''now instead of putting in some ctx[i], we're putting in the class prompt which came from the CAVPT and then do same process'''
#             if(need_attn):
#                 attn=self.resblocks[i+self.bottom_limit(x,True)]
#                 return attn

#             else:
#                 x = self.resblocks[i+self.bottom_limit](x)
#                 if n_ctx != 0:
#                     x = x[:-n_ctx, :, :]
            
#         '''FINALLY GIVING BACK THE X AFTER HAVING PASSED IT THROUGH ALL THE LAYERS OF THE VISION ENCODER BUT LIKE THIS
#         1) 11 OF THOSE LAYERS WERE DONE USING THE CONTEXT FROM THE CONTEXT LEARNER (ctx)
#         2) LAST ONE WAS DONE USING THE CLASS PROMPT FROM THE CAVPT
        
#         IN THIS CASE THE LAYER_LOGITS WILL JUST BE A SINGLE ELEMENT LONG, HAVING THE LAYER LOGITS OF THE LAST LAYER
#         '''
        
#         return x, layer_logits, attn_map


# class ImageEncoder_VPTD(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.conv1 = clip_model.visual.conv1
#         self.class_embedding = clip_model.visual.class_embedding
#         self.positional_embedding = clip_model.visual.positional_embedding
#         self.ln_pre = clip_model.visual.ln_pre
#         self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
#         self.ln_post = clip_model.visual.ln_post
#         # self.proj = clip_model.visual.proj
#         self.proj = ProjLearner(clip_model)
        
#     def forward(self, x, text_feature):
#         '''SO THE WHOLE IMAGE ALONG WITH THE TOP 10 TEXT FEATURES COME IN NOW'''
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         #x -> [32,768,7,7]
       
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         #x-> [32,49,768]

#         # class_embedding is class token.
#         '''adding the class token here'''
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         #x -> [32,50,768]
#         x = x + self.positional_embedding.to(x.dtype)
 
#         x = self.ln_pre(x)
#         #x-> [32,50,768]
        

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         #x -> [50,32,768]
#         x, layer_logits, attn_map= self.transformer(x, text_feature)
#         #x->[50,32,768]
#         #layer_logits -> list containing an element of [1,10,32,100]
#         '''THE TEXT FEATURES [TOP-10] WILL ONLY BE USED BY THE CAVPT MECHANISM'''
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         #x -> [32,50,768]

#         '''SO THIS X CAME OUT FROM ALL THE LAYERS OF THE IMAGE ENCODER, SOMETIMES WENT IN WITH THE LEARNABLE CONTEXT, SOMETIMES WITH THE CAVPT CONTEXT'''
       
#         '''AFTER ALL THESE LAYERS, WE JUST TAKE UP THE CLASS TOKEN FROM THE FEATURES, AND THEN PASS THAT ALONG THE PROJECTION LAYER'''
#         x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.
#         #x-> [32,768]
        
#         x = self.proj(x)
#         #x->[32,512]

#         '''RETURING BACK X AND THE LIST OF LAYER_LOGITS FROM THE TRANSFORMER'''
#         return x, layer_logits,attn_map


# class CustomCLIP_Selected_CoVPTDeep(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, devices):
#         super().__init__()
#         self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
#         #10
#         prompts = []
#         # for temp in IMAGENET_TEMPLATES_SELECT:
#         temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
#         prompts += [temp.format(c.replace("_", " ")) for c in classnames]
#         #so at this point, prompts is a 100 el long list of prompts like 'a photo of a {}' with the classname
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         #prompts -> [100,77], so where each of the 100 elements now get tokenized to 77 long list, where the first el will be SOS, and 
#         #then the encoding for the prompt (majorly the tokens itself) -> followed by the cls and then the <EOS>, remaining gon be 0s
       
#         clip_model.to(devices)
#         prompts = prompts.to(devices)
#         self.devices=devices
#         with torch.no_grad():
#             text_features = clip_model.encode_text(prompts)
#             #now pass in these text tokens into clip text encoder -> gives out features for the text
#             #text_features -> [100,512]
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         self.text_features = text_features
#         clip_model.to('cpu')
#         self.text_features = nn.Parameter(text_features)
#         self.classnames=classnames
#         self.cfg=cfg
#         self.clip_model=clip_model
        
#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         #this is gonna be the 77 long prompt for each class, but before embedding, so [100,77], but have the 'X X X ' instead of the learner vector
        
#         #same line just again, ignore off
#         self.text_features = nn.Parameter(text_features)

#         # visual (going to be defining 2 image encoders (VPTD and clip -> mainly for ground test))
#         self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
#         self.zeroshot_clip_image_encoder = clip_model.visual
#         # visual end (going to defining 1 itself)
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image, label=None):
#         '''So the first part in this forward is to get to know the top-10 text features which are in line with the image'''
#         '''and those indices be stored in indices'''
#         #image -> [32,3,224,224]
#         image = image.to(next(self.image_encoder.parameters()).device)
#         #what this does is sets the image to either cpu or gpu depending on what the image encoder was set to
#         #size gon remain the same [32,3,224,224]
#         with torch.no_grad():
#             zeroshotclip_image_feature = self.zeroshot_clip_image_encoder(image.type(self.dtype))
#             zeroshotclip_image_feature = zeroshotclip_image_feature / zeroshotclip_image_feature.norm(dim=-1, keepdim=True)
#             #zsclip -> [32,512]
            
#             logit_scale = self.logit_scale.exp()
#             logits = logit_scale * zeroshotclip_image_feature @ self.text_features.t()
#             #now then multiplying it with the text features, we get the logits
#             #logits -> [32,100]
#             _, indices = torch.sort(logits, descending=True)
#             #so now indices is going to have the indices of the logits sorted in descending
            
#             indices = indices[:, :self.class_prompt_num]
#             #and now we choose only the top 10 of this for every image in the batch
            
#             # mask = indices==label.unsqueeze(0).expand()
#             selected_text_features = self.text_features[indices]
        
#         '''UPTO HERE WAS TO GET THE TOP 10 FEATURES'''

#         '''NOW GETTING THE TEXT FEATURES FROM THE CONTEXT LEARNER'''
#         prompts = self.prompt_learner()
#         #[100,77,512]
#         tokenized_prompts = self.tokenized_prompts
#         #basically have the x x x instead of the learners, and not encoded
#         #[100,77]
#         '''CHANGE DONE FOR BRAIN 201 HERE
        
#         '''
#         #text_features = self.text_encoder(prompts, tokenized_prompts)
        

#         '''HERES WHAT TO DO IF YOU WANT HAND DESIGNED SET OF TEXT PROMPTS
#         #so basically untrained, and pre set context , with just the new class names will make difference here
#         #[100,512]'''
#         text_features=self.text_features

#         text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
#         #[100,512]

#         '''NOW TO GET THE IMAGE FEATURES FROM THE IMAGE ENCODER (VPTD) TO WHICH WE'LL PASS IN 2 THINGS
#         1) IMAGE [32,3,224,224]
#         2) TOP 10 TEXT FEATURES [10,512]
#         '''
#         image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features[indices])
#         #image_features->[32,512]
#         #layer_logits -> list having one element of [1,10,32,100]

#         #ONCE WE HAVE THE FINAL IMAGE FEATURES, WE JUST MULTIPLY IT WITH THE TEXT FEATURES (FULL 100, NOT TOP 10)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         logit_scale = self.logit_scale.exp()

#         #THIS ONE IS FOR VPT+COOP
#         #print('Image features', image_features.shape)
#         #print('Text features', text_features_norm.shape)
#         logits = logit_scale * image_features @ text_features_norm.t()
        
#         #THIS IS ONLY FOR COOP
#         logits1 = logit_scale * zeroshotclip_image_feature @ text_features_norm.t()
#         # loss1 = F.mse_loss(text_features, self.text_features)
#         #THIS IS ONLY FOR VPT
#         logits2 = logit_scale * image_features @ self.text_features.t()

#         #logits, logits1, logits2 -> [32,100]
        
#         #print(logits.shape,'LOGITS SHAPE')

#         return logits, layer_logits, indices, logits1, logits2 , attn_map
    

#     '''GOAL OF THIS FUNCTION IS TO USE IT FOR THE TEST PART ONLY (WHERE WE'D LIKE TO PASS IN THE LABEL OF THE CLASS TOO) -> NICELY DONE
#     USING THE label=None, AND ALSO WELL APPROACHED IN THE get_heatmap FUNCTION'''
#     def inference(self, image, label=None):
#         '''So the first part in this forward is to get to know the top-10 text features which are in line with the image'''
#         '''and those indices be stored in indices'''
#         #image -> [32,3,224,224]
#         image = image.to(next(self.image_encoder.parameters()).device)

        
#         '''HAND CRAFTER PROMPT INFERNENCE'''
        
        
        
#         temp=CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
#         prompts=[temp.format(label)]
#         prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.devices)
        
#         #print(prompts.shape)
#         text_features = self.clip_model.encode_text(prompts)
        
#         text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

#         #print('text features',text_features.shape)
        
        
#         image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features.unsqueeze(0))
        
        
        
#         '''TRAINED PROMPT INFERENCE'''
#         #print('INFERENCE USING RANDOM PROMPTS')
        
#         '''
        
#         prompts = self.prompt_learner()
#         #[2,77,512]
#         tokenized_prompts = self.tokenized_prompts
#         #[2,77]
#         text_features = self.text_encoder(prompts, tokenized_prompts)
#         #[2,512]
        
#         text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        
#         #SO BASICALLY PASSING IN ONLY THE TEXT FEATURE OF THE TUMOR TEXT, BUT NEED TO BE A 3D VECTOR, SO UNSQUEEZING AS WELL
#         image_features, layer_logits, attn_map = self.image_encoder(image.type(self.dtype), text_features[1].unsqueeze(0).unsqueeze(0))
        
#         '''
        
#         '''COMMON PART STARTS AFTER THIS'''
        
#         #image_features->[32,512]
#         #layer_logits -> list having one element of [1,10,32,100]

#         #ONCE WE HAVE THE FINAL IMAGE FEATURES, WE JUST MULTIPLY IT WITH THE TEXT FEATURES (FULL 100, NOT TOP 10)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        
#         logit_scale = self.logit_scale.exp()
        
#         #print('Image features', image_features.shape)
#         #print('Text features', text_features_norm.shape)
#         logits = logit_scale * image_features @ text_features_norm.t()
#         #print('INFERENCE LOGITS',logits.shape)

#         #print(logits.shape,'LOGITS SHAPE')
#         return logits
    
# # end

# def get_image_heat_map_new(img, attentions, head_num=-1, token=0, model="ZeroshotCLIP"):

#     patch_size = 32 # default

#     n_heads=1

#     w_featmap = img.shape[2] // patch_size
#     h_featmap = img.shape[1] // patch_size


#     if(head_num < 0):
#         attentions = attentions.reshape(1, w_featmap, h_featmap).mean(dim=0)
#     else:
#         attentions = attentions.reshape(1, w_featmap, h_featmap)[head_num]

#     attention = np.asarray(Image.fromarray((attentions*255).detach().numpy().astype(np.uint8)).resize((h_featmap * patch_size, w_featmap * patch_size))).copy()
   
#     to_pil = transforms.ToPILImage()
#     pil_image = to_pil(img)

#     #print('pil image',pil_image.size)
    
#     #print('attention', attention.shape)

#     mask = cv2.resize(attention / attention.max(), pil_image.size)[..., np.newaxis]
#     #print('mask shape',mask.shape)

#     result = (mask * pil_image).astype("uint8")

#     return result,mask


# @TRAINER_REGISTRY.register()
# class DPT(TrainerX):
#     """Context Optimization (CoOp).

#     Learning to Prompt for Vision-Language Models
#     https://arxiv.org/abs/2109.01134
#     """
    
#     def model_inference(self, input):
#         return self.model(input)[0]

#     def check_cfg(self, cfg):
#         assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames
#         self.classnames = classnames
#         self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
#         self.pretrain_c = cfg.PRETRAIN.C
#         self.alpha = cfg.TRAINER.ALPHA

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
        
#         if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         # ================================== #
#         #              VPT DEEP              #
#         # ================================== #
#         print("Building custom CLIP VPT Deep")
#         self.model = CustomCLIP_Selected_CoVPTDeep(cfg, classnames, clip_model, self.device)

#         print("Turning off gradients in both the image and the text encoder")
#         for name, param in self.model.named_parameters():
#             if "image_encoder.transformer.ctx_learner" not in name and 'extractor' not in name and "prompt_learner" not in name:
#                 param.requires_grad_(False)
#             else:
#                 print(name)
    
        
        

#         self.model.to(self.device)
#         opt_cfg = cfg.OPTIM.clone()
#         opt_cfg.defrost()
#         if cfg.DATASET.NAME=='ImageNet':
#             opt_cfg.WARMUP_EPOCH=1
#         opt_cfg.freeze()
#         # NOTE: only give prompt_learner to the optimizer
#         self.optim = build_optimizer(self.model.image_encoder.transformer.ctx_learner, opt_cfg)
#         self.sched = build_lr_scheduler(self.optim, opt_cfg)
#         self.optim1 = build_optimizer(self.model.image_encoder.transformer.extractor, opt_cfg)
#         self.sched1 = build_lr_scheduler(self.optim1, opt_cfg)
#         opt_cfg = cfg.OPTIM.clone()
#         opt_cfg.defrost()
#         # opt_cfg.WARMUP_TYPE="constant"
#         opt_cfg.WARMUP_EPOCH=1
#         opt_cfg.LR = 0.002
#         opt_cfg.freeze()
#         self.optim2 = build_optimizer(self.model.prompt_learner, opt_cfg)
#         self.sched2 = build_lr_scheduler(self.optim2, opt_cfg)
#         self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)
#         self.register_model("image_encoder.transformer.extractor", self.model.image_encoder.transformer.extractor, self.optim1, self.sched1)
#         self.register_model("prompt_learner", self.model.prompt_learner, self.optim2, self.sched2)

  

#         self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)
#         #print('img shape ',label.shape)
#         #image -> [32,3,224,224]
#         #label -> [32]
#         prec = self.cfg.TRAINER.COOP.PREC

#         #NOT GONNA BE OUR CASE
#         if prec == "amp":
#             with autocast():
#                 output = self.model(image)
#                 loss = F.cross_entropy(output, label)
#             self.optim.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optim)
#             self.scaler.update()
       
#         #OUR DEAL RIGHT HERE LAD
#         else:
            
#             real_label = label
#             #print(self.model(image)[0].shape)
#             #print(self.model(image)[1].shape)
#             #print(self.model(image)[2].shape)
#             #print(self.model(image)[3].shape)
#             #print(self.model(image)[4].shape)
#             #output, layer_logits, indices, output1, output2 = self.model(image)
            
#             output=self.model(image)[0]
#             #print('OUTPUT',output)[0]
#             layer_logits=self.model(image)[1]
#             indices=self.model(image)[2]
#             output1=self.model(image)[3]
#             output2=self.model(image)[4]
            
#             #output -> [32,100]
#             #log -> basically list of 1 element, but el having shape of [1,10,32,100], so in a crude sense [1,1,10,32,100]
#             #indices -> [32,10]
#             #output1 ->[32,100]
#             #output2 -> [32,100]


#             # indices = batch, class_prompt_num
#             # label = batch

#             '''LABEL IS BASICALLY GOING TO BE THE TOKEN OF THE RIGHT CLASS, AND WE CAN USE THE F.cross_entropy TO FIND THE ENTROPY LOSS IN THIS CASE'''
#             if self.epoch < self.pretrain_c: # output1=text, output2=image
#                 loss = F.cross_entropy(output1, label) + F.cross_entropy(output2, label) + 0.1*F.cross_entropy(output, label)
#             else:
#                 loss = F.cross_entropy(output, label)

#             '''THIS SECTION ONWARDS, THEY ARE ALSO ACCOUNTING FOR ANOTHER LOSS FUNCTION, WHICH GETS ADDED TO THE LOSS FUNCTION ABOVE
#             NOT REALLY REQUIRED AT THIS POINT OF TIME, SO NOT GETTING DEEPER INTO THIS DRILL'''
            
#             layers = len(layer_logits)
#             layer_logits = torch.cat(layer_logits, dim=0).permute(2, 0, 1, 3).reshape([-1, len(self.classnames)]) # batch, layer, class_prompt_num, class_num
#             batch_target = torch.tensor([1/self.num_classes] * len(self.classnames), dtype=torch.float16).unsqueeze(0).expand(layer_logits.shape[0], -1).to(self.device)
#             #print('layer_logits 858',layer_logits.shape)
#             #print('batch_target 859',batch_target.shape)
            
#             label = label.reshape([-1, 1]).expand(-1, self.class_prompt_num)
#             tmp = label == indices # batch, class_prompt_num
#             tmp = tmp.unsqueeze(1).expand(-1, layers, -1).reshape([-1])
#             label = label.unsqueeze(1).expand(-1, layers, -1)
#             one_hot_code = F.one_hot(label.reshape([-1]), len(self.classnames))
#             # 
#             tmp = tmp.unsqueeze(1).expand(-1, len(self.classnames))
#             one_hot_code[tmp==False] = 0
            
#             batch_target[tmp] = 0
#             batch_target = batch_target+one_hot_code
            
#             layer_logits = layer_logits[tmp]
#             batch_target = one_hot_code[tmp].to(torch.float16)
            
#             if self.class_prompt_num != 0 and layer_logits.shape != torch.Size([0]):
#                 loss = loss + self.alpha * F.cross_entropy(layer_logits.reshape([-1, self.num_classes]), batch_target.reshape([-1, self.num_classes]))
#             self.model_backward_and_update(loss)

#         loss_summary = {
#             "loss": loss.item(),
#             # "tmp": tmp.item(),
#             "acc": compute_accuracy(output, real_label)[0].item(),
#         }

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary

#     def parse_batch_train(self, batch):
#         input = batch["img"]
#         label = batch["label"]
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return

#         names = self.get_model_names()

#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         for name in names:
#             model_path = osp.join(directory, name, model_file)

#             if not osp.exists(model_path):
#                 raise FileNotFoundError('Model not found at "{}"'.format(model_path))

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]
#             epoch = checkpoint["epoch"]

#             # Ignore fixed token vectors
#             if "token_prefix" in state_dict:
#                 del state_dict["token_prefix"]

#             if "token_suffix" in state_dict:
#                 del state_dict["token_suffix"]
            
#             if 'classfiers.0.weight' in state_dict:
#                 for i in range(12 - self.cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT+1):
#                     del state_dict[f"classfiers.{i}.weight"]
#                 # del state_dict["classfiers.0.bias"]

#             print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)


#     def train(self):
#         """Generic training loops."""

#         self.before_train()
#         for self.epoch in range(self.start_epoch, self.max_epoch):
#             self.before_epoch()
#             self.run_epoch()
#             self.after_epoch()
#         self.after_train()

#     def create_maps(self,path):
    
#         '''SO BASICALLY IN BRAIN-301, AFTER BATCH NO 109 WE'D HAVE THE TUMOR IMAGES (NO SHUFFLING DONE HERE, SINCE ITS A TESTING SET'''
        
#         '''batch_nums=[0,30,60,90,120,150,180,210]
#         img_num=30 #just some random constant
        
#         batch_num=210
        
#         c=0
#         for i in self.test_loader:
#           if(c==batch_num):
#             img=i['img'][img_num]
#             print(img.shape)
#             label=i['label'][img_num]
#             if(label==0):
#                 brain_class='No Tumour'
#             else:
#                 brain_class='Tumour'    
#             torch.save(img,f'{batch_num}-{img_num}-{label}_Image.pt')
#             _=self.model.inference(torch.unsqueeze(img,dim=0),brain_class)
#           c+=1 
#         print('done')'''
        
        

#         folder_path1 = f'inference/{path}/pred'
#         folder_path2 = f'inference/{path}/img'
#         # folder_path3 = f'inference/{path}/mask'
        
#         data=path.split('/')[0]
#         shots=path.split('/')[1]
#         cat=path.split('/')[2]
        
#         if os.path.exists(folder_path1):
#           shutil.rmtree(folder_path1)
          
#           if not os.path.exists(f'inference/{data}'):
#             os.mkdir(f'inference/{data}')
          
#           if not os.path.exists(f'inference/{data}/{shots}'):
#             os.mkdir(f'inference/{data}/{shots}')
          
#           if not os.path.exists(f'inference/{data}/{shots}/{cat}'):
#             os.mkdir(f'inference/{data}/{shots}/{cat}')
            
          
#           os.mkdir(f'inference/{data}/{shots}/{cat}/pred')
#         else:
          
#           if not os.path.exists(f'inference/{data}'):
#             os.mkdir(f'inference/{data}')
          
#           if not os.path.exists(f'inference/{data}/{shots}'):
#             os.mkdir(f'inference/{data}/{shots}')
          
#           if not os.path.exists(f'inference/{data}/{shots}/{cat}'):
#             os.mkdir(f'inference/{data}/{shots}/{cat}')
            
#           os.mkdir(f'inference/{data}/{shots}/{cat}/pred')
          
          
#         if os.path.exists(folder_path2):
#           shutil.rmtree(folder_path2)
          
#           os.mkdir(f'inference/{data}/{shots}/{cat}/img')
#         else:
          
#           os.mkdir(f'inference/{data}/{shots}/{cat}/img')
          
#         # if os.path.exists(folder_path3):
#         #   shutil.rmtree(folder_path3)
          
#           # os.mkdir(f'inference/{data}/{shots}/{cat}/mask')
#         # else:
          
#         #   os.mkdir(f'inference/{data}/{shots}/{cat}/mask')
        
        
#         batch_num=0

#         for batch in self.test_loader:
#             ## BS::
#             # 302 BraTS20 ==> A-18172 + N-24265 == 848 - 35
#             # 402 BraTS21 ==> A-81241 + N-75134 == 3127 - 25
#             # BraTS23 ==> 
#             # ATLAS ==> A-32989 + N-90806 == 2475 - 45
#             if(batch_num!=3127):  # 
#                 for img_num in range(50):
                  
#                     #print('im in',img_num,batch_num)
                    
                    
#                     img=batch['img'][img_num]
#                     label=batch['label'][img_num]
#                     impath=batch['impath'][img_num]
                    
#                     if(label==1):
#                       _=self.model.inference(torch.unsqueeze(img,dim=0),'Unhealthy_Liver') 
                      
#                       file_path=impath.split('/')[-1].split('.')[0]
#                       #file_path=impath.split('/')[-1]
                      
#                       transform = transforms.ToPILImage()
#                       image = transform(img)
#                       image.save(f'{folder_path2}/{file_path}.png')
#                       #image.save(f'{folder_path2}/{file_path}')
                      
#                       attentions=torch.load('Attn_map.pt')
#                       attentions = attentions[0,0,1:50]
                      
#                       result,mask_pred=get_image_heat_map_new(img,attentions)
#                       mask_pred=torch.tensor(mask_pred).permute(2,0,1)
#                       mask_pred = transform(mask_pred)
#                       mask_pred.save(f'{folder_path1}/{file_path}.png')
#                       #mask_pred.save(f'{folder_path1}/{file_path}')
                      
#                       # mask_file=f'mask-302/{file_path}.png'
#                       # #mask_file=f'mask-601/{file_path}'
#                       # mask = Image.open(mask_file)
#                       # mask.save(f'{folder_path3}/{file_path}.png')
#                       # #mask.save(f'{folder_path3}/{file_path}')
#                       # transform = transforms.ToTensor()
#                       # mask = transform(mask)
                      
     
        
#             else:
#                 for img_num in range(25):
                
#                     #print('im in',img_num,batch_num)
                    
#                     img=batch['img'][img_num]
#                     label=batch['label'][img_num]
#                     impath=batch['impath'][img_num]
                    
#                     if(label==1):
#                       _=self.model.inference(torch.unsqueeze(img,dim=0),'Unhealthy_Liver') 
                      
#                       file_path=impath.split('/')[-1].split('.')[0]
#                       #file_path=impath.split('/')[-1]
                      
#                       transform = transforms.ToPILImage()
#                       image = transform(img)
#                       image.save(f'{folder_path2}/{file_path}.png')
#                       #image.save(f'{folder_path2}/{file_path}')
                      
#                       attentions=torch.load('Attn_map.pt')
#                       attentions = attentions[0,0,1:50]
                      
#                       result,mask_pred=get_image_heat_map_new(img,attentions)
#                       mask_pred=torch.tensor(mask_pred).permute(2,0,1)
#                       mask_pred = transform(mask_pred)
#                       mask_pred.save(f'{folder_path1}/{file_path}.png')
#                       #mask_pred.save(f'{folder_path1}/{file_path}')
                      
#                       # mask_file=f'mask-302/{file_path}.png'
#                       # #mask_file=f'mask-601/{file_path}'
#                       # mask = Image.open(mask_file)
#                       # mask.save(f'{folder_path3}/{file_path}.png')
#                       # #mask.save(f'{folder_path3}/{file_path}')
#                       # transform = transforms.ToTensor()
#                       # mask = transform(mask)
                      
            
#             batch_num+=1
            
#         #85 - 302
#         #848 - 302 full
#         #312 - 401
#         #3128 - 401 full
#         #45 -501 FULL
#         #658 FOR 601 FULL
        
#             print(f'BATCHES DONE {batch_num}/3127')
            
    
#     def generate_metric(self,folder,threshold,median):
          
#       big_img_num=0
      
#       batch_num=0
      
#       total_dice_score_4=0
#       total_dice_score_5=0
#       total_dice_score_6=0
      
#       total_auprc_4=0
#       total_auprc_5=0
#       total_auprc_6=0
      
#       total_iou_4=0
#       total_iou_5=0
#       total_iou_6=0
      
#       total_f1_4=0
#       total_f1_5=0
#       total_f1_6=0
      
#       total_recall_4=0
#       total_recall_5=0
#       total_recall_6=0
      
#       total_prec_4=0
#       total_prec_5=0
#       total_prec_6=0
      
#       big_history={}
      
#       zero_images=0
        
#       total_tumor_images=0
      
#       print('SETTING THRESHOLD FOR GRAOUND MASKS AS',threshold)
      
#       for batch in self.test_loader:
      
       
        
#         if(batch_num!=3127):  
#           for img_num in range(50):
#             impath=batch['impath'][img_num]
#             label=batch['label'][img_num]
            
#             if(label==1):
                
#                 metrics=eval_metrics(impath,folder,threshold,median)
                
#                 if(metrics==False):
#                   print('continuing')
#                   zero_images+=1
#                   continue
                
#                 big_history[impath]=metrics
                
#                 total_dice_score_4+=metrics['dice-0.4']
#                 total_dice_score_5+=metrics['dice-0.5']
#                 total_dice_score_6+=metrics['dice-0.6']
                
#                 total_iou_4+=metrics['iou-0.4']
#                 total_iou_5+=metrics['iou-0.5']
#                 total_iou_6+=metrics['iou-0.6']
                
#                 total_auprc_4+=metrics['auprc-0.4']
#                 total_auprc_5+=metrics['auprc-0.5']
#                 total_auprc_6+=metrics['auprc-0.6']
                
#                 total_f1_4+=metrics['f1-0.4']
#                 total_f1_5+=metrics['f1-0.5']
#                 total_f1_6+=metrics['f1-0.6']
                
#                 total_recall_4+=metrics['recall-0.4']
#                 total_recall_5+=metrics['recall-0.5']
#                 total_recall_6+=metrics['recall-0.6']
                
#                 total_prec_4+=metrics['precision-0.4']
#                 total_prec_5+=metrics['precision-0.5']
#                 total_prec_6+=metrics['precision-0.6']
                
                
#                 total_tumor_images+=1
                
#         else:
        
#           for img_num in range(25):
          
#             impath=batch['impath'][img_num]
#             label=batch['label'][img_num]
            
             
#             if(label==1):
               
#                 metrics=eval_metrics(impath,folder,threshold,median)
                
#                 if(metrics==False):
#                   print('continuing')
#                   zero_images+=1
#                   continue
                
#                 big_history[impath]=metrics
                
#                 total_dice_score_4+=metrics['dice-0.4']
#                 total_dice_score_5+=metrics['dice-0.5']
#                 total_dice_score_6+=metrics['dice-0.6']
                
#                 total_iou_4+=metrics['iou-0.4']
#                 total_iou_5+=metrics['iou-0.5']
#                 total_iou_6+=metrics['iou-0.6']
                
#                 total_auprc_4+=metrics['auprc-0.4']
#                 total_auprc_5+=metrics['auprc-0.5']
#                 total_auprc_6+=metrics['auprc-0.6']
                
#                 total_f1_4+=metrics['f1-0.4']
#                 total_f1_5+=metrics['f1-0.5']
#                 total_f1_6+=metrics['f1-0.6']
                
#                 total_recall_4+=metrics['recall-0.4']
#                 total_recall_5+=metrics['recall-0.5']
#                 total_recall_6+=metrics['recall-0.6']
                
#                 total_prec_4+=metrics['precision-0.4']
#                 total_prec_5+=metrics['precision-0.5']
#                 total_prec_6+=metrics['precision-0.6']
                
#                 total_tumor_images+=1
                
                

#         batch_num+=1
        
        
#         print(f'BATCHES DONE {batch_num}/485 ')
      
#       #JUST TO REMOVE OUT THE ZERO ERROR
#       if(total_tumor_images==0):
#         total_tumor_images+=1
      
      
#       with open('data.json', 'w') as f:
#         json.dump(big_history, f, indent=4)
      
#       folder=folder.replace('/','-')
      
#       if(median):
#         save_path=f'result-{folder}-{threshold}-median.txt'
#       else:
#         save_path=f'result-{folder}-{threshold}.txt'
      
      
#       with open(save_path, "w") as file:
#         file.write(f'TOTAL TUMOR IMAGES {total_tumor_images} \n')
#         file.write(f'NEGLECT IMAGES {zero_images} \n')
      
#         file.write(f'AVG DICE SCORE- 0.4 {total_dice_score_4/total_tumor_images} \n')
#         file.write(f'AVG DICE SCORE- 0.5 {total_dice_score_5/total_tumor_images} \n')
#         file.write(f'AVG DICE SCORE- 0.6 {total_dice_score_6/total_tumor_images} \n')
        
#         file.write(f'AVG IOU - 0.4 {total_iou_4/total_tumor_images} \n')
#         file.write(f'AVG IOU - 0.5 {total_iou_5/total_tumor_images} \n')
#         file.write(f'AVG IOU - 0.6 {total_iou_6/total_tumor_images} \n')
        
#         file.write(f'AVG AUPRC - 0.4 {total_auprc_4/total_tumor_images} \n')
#         file.write(f'AVG AUPRC - 0.5 {total_auprc_5/total_tumor_images} \n')
#         file.write(f'AVG AUPRC - 0.6 {total_auprc_6/total_tumor_images} \n')
        
#         file.write(f'AVG F1 - 0.4 {total_f1_4/total_tumor_images} \n')
#         file.write(f'AVG F1 - 0.5 {total_f1_5/total_tumor_images} \n')
#         file.write(f'AVG F1 - 0.6 {total_f1_6/total_tumor_images} \n')
        
#         file.write(f'AVG RECALL - 0.4 {total_recall_4/total_tumor_images} \n')
#         file.write(f'AVG RECALL - 0.5 {total_recall_5/total_tumor_images} \n')
#         file.write(f'AVG RECALL - 0.6 {total_recall_6/total_tumor_images} \n')
        
#         file.write(f'AVG PRECISION - 0.4 {total_prec_4/total_tumor_images} \n')
#         file.write(f'AVG PRECISION - 0.5 {total_prec_5/total_tumor_images} \n')
#         file.write(f'AVG PRECISION - 0.6 {total_prec_6/total_tumor_images} \n')
       
#       print(f'DONE SAVING {save_path}')

    
    
      
#     def check(self, fold,path):
    
#       #THIS IS FOR THE CLASSIFICATION DEAL
      
#       c1=0
#       c2=0
      
#       tumor_pred=0
#       tumor_actual=0
#       batch_num=0
      
#       #FROM GENERATE METRICS
      
#       big_img_num=0
#       batch_num=0
      
#       total_dice_score_4=0
#       total_dice_score_5=0
#       total_dice_score_6=0
      
#       total_auprc_4=0
#       total_auprc_5=0
#       total_auprc_6=0
      
#       total_iou_4=0
#       total_iou_5=0
#       total_iou_6=0
      
#       total_f1_4=0
#       total_f1_5=0
#       total_f1_6=0
      
#       total_recall_4=0
#       total_recall_5=0
#       total_recall_6=0
      
#       total_prec_4=0
#       total_prec_5=0
#       total_prec_6=0
      
#       big_history={}
      
#       zero_images=0
        
#       total_tumor_images=0
      
#       #folder='302-501Full/10000-100/1'
#       folder=path
      
#       threshold=0
      
#       for batch in self.test_loader:
        
#         if(batch_num!=3127):  
#           for img_num in range(50):
          
#             img=batch['img'][img_num]
#             label=batch['label'][img_num]
#             impath=batch['impath'][img_num]
          
#             if(label==1): 
#               tumor_actual+=1 
#               out=self.model(torch.unsqueeze(img,dim=0),'Tumour') 
#               #print(out)
#               logits=out[0]
              
              
#               if(logits[0][0]<logits[0][1]):
              
#                 tumor_pred+=1
                
#                 metrics=eval_metrics(impath,folder,threshold)
                
#                 if(metrics==False):
#                   print('continuing')
#                   zero_images+=1
#                   continue
                
#                 big_history[impath]=metrics
                
#                 total_dice_score_4+=metrics['dice-0.4']
#                 total_dice_score_5+=metrics['dice-0.5']
#                 total_dice_score_6+=metrics['dice-0.6']
                
#                 total_iou_4+=metrics['iou-0.4']
#                 total_iou_5+=metrics['iou-0.5']
#                 total_iou_6+=metrics['iou-0.6']
                
#                 total_auprc_4+=metrics['auprc-0.4']
#                 total_auprc_5+=metrics['auprc-0.5']
#                 total_auprc_6+=metrics['auprc-0.6']
                
#                 total_f1_4+=metrics['f1-0.4']
#                 total_f1_5+=metrics['f1-0.5']
#                 total_f1_6+=metrics['f1-0.6']
                
#                 total_recall_4+=metrics['recall-0.4']
#                 total_recall_5+=metrics['recall-0.5']
#                 total_recall_6+=metrics['recall-0.6']
                
#                 total_prec_4+=metrics['precision-0.4']
#                 total_prec_5+=metrics['precision-0.5']
#                 total_prec_6+=metrics['precision-0.6']
               
                
#               if(logits[0][1]<logits[0][0]):
                
#                   print('MODEL PREDICTING 0 EVEN THOUGH ITS UN-HEALTHY')
#                   c2+=1
            
        
#         else:
#           for img_num in range(25):
          
#             img=batch['img'][img_num]
#             label=batch['label'][img_num]
            
#             if(label==1): 
#               tumor_actual+=1 
#               out=self.model(torch.unsqueeze(img,dim=0),'Tumour') 
#               logits=out[0]
              
#               if(logits[0][0]<logits[0][1]):
              
#                 tumor_pred+=1
                
#                 metrics=eval_metrics(impath,folder,threshold)
                
#                 if(metrics==False):
#                   print('continuing')
#                   zero_images+=1
#                   continue
                
#                 big_history[impath]=metrics
                
#                 total_dice_score_4+=metrics['dice-0.4']
#                 total_dice_score_5+=metrics['dice-0.5']
#                 total_dice_score_6+=metrics['dice-0.6']
                
#                 total_iou_4+=metrics['iou-0.4']
#                 total_iou_5+=metrics['iou-0.5']
#                 total_iou_6+=metrics['iou-0.6']
                
#                 total_auprc_4+=metrics['auprc-0.4']
#                 total_auprc_5+=metrics['auprc-0.5']
#                 total_auprc_6+=metrics['auprc-0.6']
                
#                 total_f1_4+=metrics['f1-0.4']
#                 total_f1_5+=metrics['f1-0.5']
#                 total_f1_6+=metrics['f1-0.6']
                
#                 total_recall_4+=metrics['recall-0.4']
#                 total_recall_5+=metrics['recall-0.5']
#                 total_recall_6+=metrics['recall-0.6']
                
#                 total_prec_4+=metrics['precision-0.4']
#                 total_prec_5+=metrics['precision-0.5']
#                 total_prec_6+=metrics['precision-0.6']
                 
#               if(logits[0][1]<logits[0][0]):
                
#                   print('MODEL PREDICTING 0 EVEN THOUGH ITS UN-HEALTHY')
#                   c2+=1
              
              
        
#         batch_num+=1
        
#         #85 -   302
#         #848 -  302 FULL
#         #312 -  401
#         #3128 - 401 FULL
#         #45 -   501 FULL
#         #660 -  601 FULL
        
#         print(f'{batch_num}/485')
        
#       print(c1,c2)
#       print('Tumour ACTUAL',tumor_actual)
#       print('Tumour pred',tumor_pred)
      
#       with open('data.json', 'w') as f:
#         json.dump(big_history, f, indent=4)
      
#       folder=folder.replace('/','-')
      
#       file_name=f'10000_BraTS20_BraTS20-{fold}.txt'
#       with open(file_name, "w") as file:
#         file.write(f'TOTAL TUMOR IMAGES {tumor_actual} \n')
#         file.write(f'PREDICTED {tumor_pred} \n')
      
#         file.write(f'AVG DICE SCORE- 0.4 {total_dice_score_4/tumor_pred} \n')
#         file.write(f'AVG DICE SCORE- 0.5 {total_dice_score_5/tumor_pred} \n')
#         file.write(f'AVG DICE SCORE- 0.6 {total_dice_score_6/tumor_pred} \n')
        
#         file.write(f'AVG IOU - 0.4 {total_iou_4/tumor_pred} \n')
#         file.write(f'AVG IOU - 0.5 {total_iou_5/tumor_pred} \n')
#         file.write(f'AVG IOU - 0.6 {total_iou_6/tumor_pred} \n')
        
#         file.write(f'AVG AUPRC - 0.4 {total_auprc_4/tumor_pred} \n')
#         file.write(f'AVG AUPRC - 0.5 {total_auprc_5/tumor_pred} \n')
#         file.write(f'AVG AUPRC - 0.6 {total_auprc_6/tumor_pred} \n')
        
#         file.write(f'AVG F1 - 0.4 {total_f1_4/tumor_pred} \n')
#         file.write(f'AVG F1 - 0.5 {total_f1_5/tumor_pred} \n')
#         file.write(f'AVG F1 - 0.6 {total_f1_6/tumor_pred} \n')
        
#         file.write(f'AVG RECALL - 0.4 {total_recall_4/tumor_pred} \n')
#         file.write(f'AVG RECALL - 0.5 {total_recall_5/tumor_pred} \n')
#         file.write(f'AVG RECALL - 0.6 {total_recall_6/tumor_pred} \n')
        
#         file.write(f'AVG PRECISION - 0.4 {total_prec_4/tumor_pred} \n')
#         file.write(f'AVG PRECISION - 0.5 {total_prec_5/tumor_pred} \n')
#         file.write(f'AVG PRECISION - 0.6 {total_prec_6/tumor_pred} \n')
       
#       print(file_name)
