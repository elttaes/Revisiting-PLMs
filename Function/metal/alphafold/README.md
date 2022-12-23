If you want to finetune alphafold, you just need "python train.py".But there are some things you need to modify

1. Dir "model_params_dir" should include the "params" folder, which includes the original Alphafold params(params_model_1/2/3/4/5.npz).

2. Because we using the "pmap", the GPU number should >=2. If you do not have multiple GPUs, "state = jax.pmap(updater.init)(rng_pmap, data)" should be changed into "state = updater.init(rng, data)"

3. You need modify "features_dir" in function "datasets_train/test"
