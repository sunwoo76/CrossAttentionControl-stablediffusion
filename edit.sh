python ./scripts/editimg.py\
    --prompt "A man wearing a red suit"\
    --init-img "/root/project/blended-diffusion/input_example/bld_man/bld_man.jpg"\
    --init-mask "/root/project/blended-diffusion/input_example/bld_man/bld_man_mask_2.png"\
    --strength 0.99\
    --n_samples 3\
    --use_mask\
    # --save_attn_dir "/root/media/data1/sdm/attenmaps_redsuit_22/"\
    # --is_get_attn\
    