"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from ldm.util import default

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        
        self.register_buffer('ddim_sqrt_one_minus_alphas_prev', np.sqrt(1. - ddim_alphas_prev))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    # def q_sample(self, x_start, t, noise=None):
    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     return (extract_into_tensor(self.ddim_alphas, t, x_start.shape) * x_start +
    #             extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x_start.shape) * noise)

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def q_sample_ddim_prev(self, x, index, device, noise=None):
        b,c,h,w = x.shape

        noise = default(noise, lambda: torch.randn_like(x))

        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas_prev = self.ddim_sqrt_one_minus_alphas_prev

        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device) # (2,1,1,1)
        sqrt_mi_a_prev = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas_prev[index], device=device) # (2,1,1,1)

        return a_prev*x + sqrt_mi_a_prev*noise

    @torch.no_grad()
    def ddim_reparam(self, x, e_t, sqrt_one_minus_at, a_t, a_prev, sigma_t, temperature, device, repeat_noise, noise_dropout, quantize_denoised):
        # current prediction for x_0
        """ 2. eq12) left term: reparameterization trick 통해서 x_0 예측. """
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            # False.
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        """ 3. eq12) right term: direction pointing to x_t """
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        """ 4. predict x_{t-1} """
        """ static thresholding """
        # x_prev = a_prev.sqrt() * torch.clip(pred_x0, -1., 1.) + dir_xt + noise
        #print("@@ prev_x0: max:{0}, min:{1}".format(torch.max(pred_x0), torch.min(pred_x0)))
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # x_prev = a_prev.sqrt() * pred_x0.clamp(-5.4794520548, 5.4794520548) + dir_xt + noise

        return x_prev, pred_x0

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, sx=None, sc=None, pmask=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        paramter -> default arugments
        unconditional_guidance_scale=7.5

        sw: sx, sc(source condition) added, for swapping
        """

        b, *_, device = *x.shape, x.device

        """ set DDIM hyper-parameters """
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas # ( num_timesteps  =50)
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device) # (6,1,1,1)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device) # (6,1,1,1)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device) # (6,1,1,1)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device) # (6,1,1,1)

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            """ 1. input to the U-Net: get predicted noise at this time step. """
            x_in = torch.cat([x] * 2) # 6 (abc,abc)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) # (6, 77, 768) = (uc,uc,uc, c,c,c)
            
            if sc is None:
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) # classifier free guidance
            else:
                """ swapping """
                sx_in = torch.cat([sx] * 2)
                sc_in = torch.cat([unconditional_conditioning, sc]) # (6, 77, 768) # source condition이 있다면 같이 넣어줘서 attention을 처리해줘야함.
                #                                        x_noisy, t, cond, scond=None, 
                c_out, sc_out = self.model.apply_model(x_in, t_in, c_in, sx_noisy=sx_in, scond=sc_in, pmask=pmask).chunk(2) # target, source
                e_t_uncond, e_t = c_out.chunk(2)
                se_t_uncond, se_t = sc_out.chunk(2)

                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) # classifier free guidance
                se_t = se_t_uncond + unconditional_guidance_scale * (se_t - se_t_uncond) # classifier free guidance

        #import pdb; pdb.set_trace
        # if score_corrector is not None:
        #     # None.
        #     assert self.model.parameterization == "eps"
        #     e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        x_prev, pred_x0 = self.ddim_reparam(x, e_t, sqrt_one_minus_at, a_t, a_prev, sigma_t, temperature, device, repeat_noise, noise_dropout, quantize_denoised)
        if sc is  None:
            return x_prev, pred_x0
        else:
            sx_prev, spred_x0 = self.ddim_reparam(sx, se_t, sqrt_one_minus_at, a_t, a_prev, sigma_t, temperature, device, repeat_noise, noise_dropout, quantize_denoised)
            return x_prev, pred_x0, sx_prev, spred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        """ get x_t thorugh reparameterization trick
        -> ddim sampler -> ddim encoding.
        default: use_original_steps=False, noise=None
        """
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod # sqrt(a^{bar})
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod# sqrt(1-a^{bar})
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    # sampler.decode(z_enc, c, t_enc, x0=init_latent_bg, mask=init_mask, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,)
    def decode(self, x_latent, cond, t_start, sx_latent=None, scond=None, pmask=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, x0=None, mask=None):
        """
        decode->p_sample_ddim
        default:
            x_latent.shape = (3,4,64,64) # based on resolution 512
            self.ddpm_num_timesteps=1000
            t_start=25 # timesteps[25] = 481 start from this point. (txt2img)
            unconditional_guidance_scale=5.0
            cond.shap, unconditional_conditioning.shape = (3,77,768)
            use_original_steps = False
        
        sw: sx_latent, scond, x0, mask, pmask추가.
        """
        # self.ddim_timesteps.
        """
        (Pdb) self.ddim_timesteps
                array([  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
                        261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
                        521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
                        781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981])
        """
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start] # will be fliped

        time_range = np.flip(timesteps) #[481, 461, ... , 1]
        #import pdb; pdb.set_trace()
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)

        x_dec = x_latent
        """ added for swaaping. """
        sx_dec = sx_latent

        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # total_step=25, index=24, 23, 22, ..., 0, step=481, 461, ..., 1

            # batch 수에 맞게 현재 timestep 채워줌.
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long) #shape=(2, ...)
            
            if scond is None:
                x_dec, _ = self.p_sample_ddim(x_dec, cond, ts,
                                            index=index, use_original_steps=use_original_steps,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning)
            else:
                x_dec, _, sx_dec, _ = self.p_sample_ddim(x_dec, cond, ts, sx=sx_dec, sc=scond, pmask=pmask,
                                            index=index, use_original_steps=use_original_steps,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning)   
            if mask is not None:
                assert x0 is not None
                if index != 0 :
                    qts = torch.full((x_latent.shape[0],), time_range[i+1], device=x_latent.device, dtype=torch.long) #shape=(2, ...)
                    img_orig = self.model.q_sample(x0, qts)  # TODO: deterministic forward pass?
                else:
                    img_orig = x0

                mask = mask[:, 0:1, :, :]

                x_dec =  mask * x_dec + (1. - mask) * img_orig

            # if mask is not None:
            #     assert x0 is not None
            #     #img_orig = self.model.q_sample_prev(x0, ts)  # TODO: deterministic forward pass?
            #     img_orig = self.q_sample_ddim_prev(x0, index, device=x0.device)
            #     mask = mask[:, 0:1, :, :]
            #     x_dec =  mask * x_dec + (1. - mask) * img_orig

        if scond is None:
            return x_dec
        else:
            return x_dec, sx_dec