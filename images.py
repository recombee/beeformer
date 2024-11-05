from config import config
from utils import *
import transformers
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
import math 
from PIL import Image
import os

ERR = []
TOK = []

def read_image(id, fn, path, suffix=""):
    try:
        img=Image.open(os.path.join(path,id+suffix+".jpg"))
    except FileNotFoundError:
        print(f"Error reading image {os.path.join(path,id+suffix+'.jpg')}. Replacing with empty image.")
        img=Image.fromarray(np.zeros([10,10,3]).astype('uint8'), 'RGB')
        ERR.append(id)
    except:
        print(f"Differnt error with image {os.path.join(path,id+suffix+'.jpg')}. Replacing with empty image.")
        img=Image.fromarray(np.zeros([10,10,3]).astype('uint8'), 'RGB')
    try:
        return fn([img])
    except:
        print(f"Tokenizatition error for {os.path.join(path,id+suffix+'.jpg')}")
        TOK.append(id)
        return None

def read_images_into_dict(ids, fn, path, suffix=""):
    t={}
    for id in tqdm(ids):
        t[id]=(read_image(id,fn,path,suffix))
    return t

def read_images(ids, fn, path, suffix=""):
    t=[]
    for id in tqdm(ids):
        t.append(read_image(id,fn, path))
    tt = {}
    for key in tqdm(t[0].keys()):
        if isinstance(t[0][key], list):
            tt[key]=[j for i in [x[key] for x in t] for j in i]
        else:
            tt[key]=torch.vstack([x[key] for x in t])
    
    return tt

def read_images_from_dict(ids, dictionary):
    t=[]
    for id in tqdm(ids):
        t.append(dictionary[id])
    tt = {}
    for key in tqdm(t[0].keys()):
        if isinstance(t[0][key], list):
            tt[key]=[j for i in [x[key] for x in t] for j in i]
        else:
            tt[key]=torch.vstack([x[key] for x in t])
    
    return tt

class ImageModel(torch.nn.Module):
    def __init__(self, model_name, device, pooling="CLS", trust_remote_code=False):
        super().__init__()
        self.pooling = pooling.lower()
        assert self.pooling in ("cls", "mean")
        self.model_name = model_name
        self._load(model_name, trust_remote_code)
        self.device = device
        self.to(device)

    def tokenize(self, images):
        return self.processor(images, return_tensors="pt")

    def forward(self, data):
        out = self.model(**data).last_hidden_state
        
        if self.pooling == "cls":
            out = out[:,0]
        else:
            out = out.mean(dim=1)
        
        return {'sentence_embedding': out}

    def move_tokens_to_device(self, tokens, ind_min=None, ind_max=None):
        if ind_min is not None and ind_max is not None:
            return {k: v[ind_min:ind_max].to(self.device) if isinstance(v, torch.Tensor) else v[ind_min:ind_max] for k, v in tokens.items()}
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

    def encode(self, tokenized_images, batch_size=32, show_progress_bar=False):
        l = get_first_item(tokenized_images).shape[0]
        max_i = math.ceil(l / batch_size)
        ret = []
        with torch.no_grad():
            for i in tqdm(range(max_i)):
                ind = i * batch_size
                ind_min = ind
                ind_max = ind + batch_size
                tokens_to_encode = self.move_tokens_to_device(tokenized_images, ind_min,ind_max)
                ret.append(self(tokens_to_encode)['sentence_embedding'])
            return torch.vstack(ret)

    def save(self, model_name=None):
        self.model.save_pretrained(self.model_name if model_name is None else model_name)
        self.processor.save_pretrained(self.model_name if model_name is None else model_name)

    def _load(self, model_name, trust_remote_code):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # hack for using CLIP model image encoder
        if isinstance(self.model, transformers.models.clip.modeling_clip.CLIPModel):
            print("creating clip vision model")
            self.model = CLIPVisionModel.from_pretrained(model_name)

