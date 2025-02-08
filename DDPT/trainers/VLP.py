import os.path as osp

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
from trainers.zsclip import CUSTOM_TEMPLATES

_tokenizer = _Tokenizer()
devices='cpu'
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
       
        x = prompts + self.positional_embedding.type(self.dtype)
        #x -> [100,77,512]
       

        x=x.permute(1,0,2)
        x=self.transformer(x)[0]   #taking first element since the 2nd one is basically the attn weights
       
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        #x->[100,77,512]

        # x.shape = [batch_size, n_ctx, transformer.width] hmmmm what??

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #so in this step kinda like were taking only a few features after doing the linear layer, and then passing through the projection layer

        #x-> [100,512]
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
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
        

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
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

        prefix = self.token_prefix
        #prefix -> [100,1,512] -> the SOS embeddings for all the 100 classes
        suffix = self.token_suffix
        #suffix -> [100,60,512 ] -> the CLS+EOS embeddings for all the 100 classes

        #1+60 = 61, and we have exactly 16 left out for the n_ctx (length of context for each prompt)

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
        return prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        return logits


# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  VPT DEEP!                   #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        
        ctx_vectors = torch.empty(self.layers, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        for i in range(self.layers):
            nn.init.normal_(ctx_vectors[i], std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        return self.ctx

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj
        
    def forward(self,x):
        if self.proj is not None:
            x = x @ self.proj
        return x
    
class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers

        # model
        transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers

        self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)


    def forward(self, x):
        ctx = self.ctx_learner()
        ctx = ctx.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        
        joint_attn=[]
        for i in range(self.layers): 
            if i != 0:
                x = x[:-self.n_ctx, :, :]  #basically removes off the last 10 (kind of like the ctx that we added in previously)
                #x-> [50,32,768]

           
            
            x = torch.cat([x, ctx[i]], dim=0)
            #x-> [60,32,768]
            x,attn_weights = self.resblocks[i](x)
            #x->[60,32,768]
            joint_attn.append(attn_weights)

        #return x,joint_attn
        joint_attn=torch.cat(joint_attn,dim=0)
        return x,joint_attn


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
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x,attn = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.

        x = self.proj(x)

        return x


class CustomCLIP_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model, devices):
        super().__init__()
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        print(len(prompts))
        #basically gonna be a 10
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        print(prompts.shape)
        #promtps->
        clip_model.to(devices)
        prompts = prompts.to(devices)
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            print('text',text_features.shape)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        clip_model.to('cpu')
        self.text_features = nn.Parameter(text_features)
        
        # visual
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
        # visual end
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        #so what this does is it returns a prompt vector of size [100,77,512], which means this
            #there are 100 classes
            #for which the 1st (out of 77) object is going to be the SOS token
            #the next 16 are going to be the learnable context vectors
            #the remaining 60 are the CLS, and the EOS tokens, which again not gon change much
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #so now this is going to be the prompts tokenized and not embedded like in the previous line
        #[100,77]

        self.text_encoder = TextEncoder(clip_model)
        #deal is that it takes in the learnable prompts, does a whole transformer scene, also uses the orignal tokenized prompts to
        #finally give out something of shape [100,512] -> almost just ready to multiply with the image embeddings 
        
        self.zeroshot_clip_image_encoder = clip_model.visual
        #straight up take the visual encoder from clip


    def forward(self, image):
        image = image.to(next(self.image_encoder.parameters()).device)
        #image ->[32,3,224,224]
        with torch.no_grad():
            zeroshotclip_image_feature = self.zeroshot_clip_image_encoder(image.type(self.dtype))[0] #since our clip image encoder also returns out the attention weights now
            #zsclip_img_feat -> [32,512]
            zeroshotclip_image_feature = zeroshotclip_image_feature / zeroshotclip_image_feature.norm(dim=-1, keepdim=True)
            
        prompts = self.prompt_learner()
        #prompts-> [100,77,512]
        tokenized_prompts = self.tokenized_prompts
        #tokenized_promtps -> [100,77]
        text_features = self.text_encoder(prompts, tokenized_prompts)
        #text features -> [100,512]

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        #image features -> [32,512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        #THE LOGITS ARE FOR THE FULL PROCESS [IMAGE FEAT FROM LEARNED VISION] [TEXT FEAT FROM LEARNED TEXT ENCODER]  (VPT+CLIP)
        logits1 = logit_scale * zeroshotclip_image_feature @ text_features.t()
        #THESE LOGITS ARE FOR THE ORIGINAL CLIP VISUAL AND TEXT WITH COOP
        # loss1 = F.mse_loss(text_features, self.text_features)
        logits2 = logit_scale * image_features @ self.text_features.t()
        #THESE ARE FOR TEXT FROM CLIP AND IMAGE FROM VPT

        #SO WILL BE PUTTING UP A LOSS FUNCTION BASED ON ALL THESE 3 LOGITS
        
        return logits, logits1, logits2

# end


@TRAINER_REGISTRY.register()
class VLP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def model_inference(self, input):
        return self.model(input)[0]
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # ================================== #
        #              VPT DEEP              #
        # ================================== #
        print("Building custom CLIP VPT Deep")
        self.model = CustomCLIP_VPTD(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_encoder.transformer.ctx_learner" not in name and "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
 
        self.pretrain_c = cfg.PRETRAIN.C
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        if cfg.DATASET.NAME=='ImageNet':
            opt_cfg.WARMUP_EPOCH=1
        # opt_cfg.LR = 0.002
        print(f"Wanted WARMUP_EPOCH = {opt_cfg.WARMUP_EPOCH}")
        opt_cfg.freeze()
        self.optim = build_optimizer(self.model.image_encoder.transformer.ctx_learner, opt_cfg)
        self.sched = build_lr_scheduler(self.optim, opt_cfg)
        # self.optim1 = build_optimizer(self.model.image_encoder.proj, cfg.OPTIM)
        # self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        opt_cfg.WARMUP_EPOCH=1
        opt_cfg.LR = 0.002
        opt_cfg.freeze()
        self.optim2 = build_optimizer(self.model.prompt_learner, opt_cfg)
        self.sched2 = build_lr_scheduler(self.optim2, opt_cfg)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim2, self.sched2)

        # ================END=============== #
        #                 END                #
        # ================END=============== #


        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output1, output2 = self.model(image)
           
            if self.epoch < self.pretrain_c:
                loss = F.cross_entropy(output1, label) + F.cross_entropy(output2, label) + 0.1*F.cross_entropy(output, label)
            else:
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def model_backward_and_update(self, loss, names=None):
            self.model_zero_grad(names)
            self.model_backward(loss)
            self.model_update(names)