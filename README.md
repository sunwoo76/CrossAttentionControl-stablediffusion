# CrossAttentionControl-stablediffusion
Unofficial implementation of "Prompt-to-Prompt Image Editing with Cross Attention Control" with Stable Diffusion, the code is based on the offical StableDiffusion repository.


The repository reproduced the cross attention control algorithm in "Prompt-to-Prompt Image Editing with Cross Attention Control". The code is based on the official [stable diffusion repository](https://github.com/CompVis/stable-diffusion)


# Cross Attention Control
```
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

# Get 
# Visualize Cross Attention Map
We follow the visualization cross-attention map as described in the Prompt-to-Prompt.
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

# Reference
[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)
[Compvis/stablediffusion](https://github.com/CompVis/stable-diffusion)
[Unofficial implementation of cross attention control](https://github.com/bloc97/CrossAttentionControl)



