# MPKD-DCFI

## Installation
This repo was tested with Ubuntu 16.04.6 LTS, Python 3.6. 
And it should be runnable with PyTorch versions >= 0.4.0.

## Running
1.Fetch the pretrained teacher models by: 

```
sh train_single.sh 
```
which will run the code and save the models to <code> ./run/$dataset/$seed/$model/ckpt </code>

The flags in <code>train_single.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>model</code>: specify the model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/single/$model.yml'</code>. 



2.Run our spot-adaptive KD by:
```
sh train.sh
```

3.(Optional) run the anti spot-adaptive KD by:

```
sh train_anti.sh
```

The flags in <code>train.sh</code> and <code>train_anti.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>net1</code>: specify the teacher model, see <code>'models/__init__.py'</code> to check the available model types.
- <code>net2</code>: specify the student model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/$distiller/$net1-$net2.yml'</code>. 



