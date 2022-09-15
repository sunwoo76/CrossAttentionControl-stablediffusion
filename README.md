# CrossAttentionControl-stablediffusion
Unofficial implementation of "Prompt-to-Prompt Image Editing with Cross Attention Control" with Stable Diffusion, the code is based on the offical StableDiffusion repository.


The repository reproduced the cross attention control algorithm in "Prompt-to-Prompt Image Editing with Cross Attention Control". The code is based on the official [stable diffusion repository](https://github.com/CompVis/stable-diffusion)

# Setting envirnoment
Please refer to [compvis/stablediffusion](https://github.com/CompVis/stable-diffusion) for set environment.
The repository is based on compvis/stablediffusion repository.

If you clone this repository run the commend as below:
```
conda env create -f environment.yaml
conda activate p2p
```

# Cross Attention Control
The word swapping, adding new phrase and reweighting function is implemented as below:
```
# located in "./ldm/modules/attention.py"
def cross_attention_control(self, tattmap, sattmap=None, pmask=None, t=0, tthres=0, token_idx=[0], weights=[[1. , 1. , 1.]]):
    attn = tattmap
    sattn = sattmap

    h = 8
    bh, n, d = attn.shape

    if t>=tthres:
        """ 1. swap & ading new phrase """
        if sattmap is not None:
            bh, n, d = attn.shape
            pmask, sindices, indices = pmask
            pmask = pmask.view(1,1,-1).repeat(bh, n, 1)
            attn = (1-pmask)*attn[:,:,indices] + (pmask)*sattn[:,:,sindices]

        """ 2. reweighting """
        attn = rearrange(attn,'(b h) n d -> b h n d', h=h) # (6,8,4096,77) -> (img1(uc), img2(uc), img1(c), img1(c), img2(c), img3(c))
        num_iter = bh//(h*2) #: 3
        for k in range(len(token_idx)):
            for i in range(num_iter):
                attn[num_iter+i, :, :, token_idx[k]] *= weights[k][i]
        attn = rearrange(attn,'b h n d -> (b h) n d', h=h) # (6,8,4096,77)

    return attn
```

The mask and indice are from the function in "./swap.py":
```
def get_indice(model, prompts, sprompts, device="cuda"):
    """ from cross attention control(https://github.com/bloc97/CrossAttentionControl) """
    # input_ids: 49406, 1125, 539, 320, 2368, 6765, 525, 320, 11652, 49407]
    tokenizer = model.cond_stage_model.tokenizer
    tokens_length = tokenizer.model_max_length

    tokens = tokenizer(prompts[0], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
    stokens= tokenizer(sprompts[0], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
    
    p_ids = tokens.input_ids.numpy()[0]
    sp_ids = stokens.input_ids.numpy()[0]


    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, sp_ids, p_ids).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]
    
    mask = mask.to(device)
    indices = indices.to(device)
    indices_target = indices_target.to(device) 

    return [mask, indices, indices_target]
```

# Word swapping & Adding new phrase
Run the shell script, ./swap.sh written as below:
```
python ./scripts/swap.py\
    --prompt "a cake with jelly beans decorations"\
    --n_samples 3\
    --strength 0.99\
    --sprompt "a cake with decorations"\
    --is-swap\
    #--fixed_code\
    # --save_attn_dir "/root/media/data1/sdm/attenmaps_apples_swap_orig/"\
    # --is_get_attn\

chmod -R 777 ./
```

If you want to get reulsts with only target prompt, annotate the arguments "is-swap" and "--sprompt". The final shell script is written as below:
```
python ./scripts/swap.py\
    --prompt "a cake with jelly beans decorations"\
    --n_samples 3\
    --strength 0.99\
    #--sprompt "a cake with decorations"\
    #--is-swap\
    #--fixed_code\
    # --save_attn_dir "/root/media/data1/sdm/attenmaps_apples_swap_orig/"\
    # --is_get_attn\

chmod -R 777 ./
```

The results are save in "./outputs/swap-samples"

# Reweighting
The reweighting function is implemented, but it can't be controlled by argument. The weight contorll through argument is not yet implemented.

Therefore, you should changed the weight for the specific token index as below:
```
The code is located on the line245 and 252 in "./ldm/modules/attenion.py"

def forward(self, x, context=None, scontext=None, pmask=None, time=None, mask=None):
    """
    x.shape: (6,4096,320)
    context.shape(6,77,768)
    q, k, v shape: (6, hw, 320), (6, 77, 320), (6, 77, 320)
    -> q,k,v shape: (32, hw, 40=320/8=self.head), (32, 77, 40=320/8=self.head), (32, 77, 40=320/8=self.head)

    - visualization.
    1. aggregate all attention map across the "timesteps" and "heads"
    2. Normalization divided by "max" with respecto to "each token"
    """

    h = self.heads
    if scontext == "selfattn":
        sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
        sattn = None
    else:
        if scontext is None:
            sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
            sattn = None

            """ cross attention control: only reweighting is possible. """
            """ The swap and adding new phrase do not work because, the source prompt does not exist in this case. """
            attn = self.cross_attention_control(tattmap=attn, t=time, token_idx=[2], weights=[[-2., 1., 5.]] )
        else:
            x, sx = x.chunk(2)
            sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
            ssim, sattn, sv = self.get_attmap(x=sx, h=self.heads, context=scontext, mask=None)

            """ cross attention control """
            bh, hw, tleng = attn.shape
            attn = self.cross_attention_control(tattmap=attn, sattmap=sattn, pmask=pmask, t=time, token_idx=[0], weights=[[1., 1., 1.]] )
```

# Visualize Cross Attention Map
We follow the visualization cross-attention map as described in the Prompt-to-Prompt:
```
def avg_attmap(self, attmap, token_idx=0):
    """
    num_sample(=batch_size) = 3
    uc,c = 2 #(unconditional, condiitonal)
    -> 3*2=6

    attmap.shape: similarity matrix.
    token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
    """
    nsample2, head, hw, context_dim = attmap.shape

    #import pdb; pdb.set_trace()
    attmap_sm = F.softmax(attmap.float(), dim=-1)#F.softmax(torch.Tensor(attmap).float(), dim=-1) # (6, 8, hw, context_dim)
    att_map_sm = attmap_sm[nsample2//2:, :, :, :] # (3, 8, hw, context_dim)
    att_map_mean = torch.mean(att_map_sm, dim=1) # (3, hw, context_dim)

    b, hw, context_dim = att_map_mean.shape
    h = int(math.sqrt(hw))
    w = h

    return att_map_mean.view(b,h,w,context_dim)  # (3, h, w, context_dim)
```

# To do.
* Implementation of controlling reweighting function through argument.
* Any resolution inference: The code is now operated in only the resolution 512x512. Some parts are hard-coded the resolution of images. 

# Reference
[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)  
[Compvis/stablediffusion](https://github.com/CompVis/stable-diffusion)  
[Unofficial implementation of cross attention control](https://github.com/bloc97/CrossAttentionControl)  



