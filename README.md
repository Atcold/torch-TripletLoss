# TripletEmbedding Criterion

This aims to reproduce the loss function used in Google's [FaceNet paper](http://arxiv.org/abs/1503.03832v1).

```lua
criterion = nn.TripletEmbeddingCriterion([alpha])
```

The cost function can be expressed as follow

![                 1  __                       2                            2 
L({a, p, n})  =  - \     max(0, ||a   -  p ||   +  alpha  -  ||a   -  n || )
                 N /__ i           i      i                     i      i    
](https://latex.codecogs.com/svg.latex?L%28%7Ba%2C%20p%2Cn%7D%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_i%20%5Cmax%280%2C%20%7C%7Ca_i%20-%20p_i%7C%7C%5E2%20&plus;%20%5Calpha%20-%20%7C%7Ca_i%20-%20n_i%7C%7C%5E2%29)


where `a`, `p` and `n` are batches of the embedding of *ancore*, *positive* and *negative* samples respectively.

If the margin `alpha` is not specified, it is set to `0.2` by default.

## Test

In order to test the criterion, someone can run the [`test`](test.lua) script as

```lua
th test.lua
```

which shows how to use the criterion and checks the correctness of the gradient.

## Training

The folder [`xmp`](xmp) contains two examples which show how a network can be trained with this criterion.

 - [`recycle-embedding`](xmp/recycle-embedding.lua) recycles the embedding of the *positive* and *negative* sample from the previous epoch (faster training, less accurate)
 - [`fresh-embedding`](xmp/fresh-embedding.lua) computes the updated embedding of all *ancore*, *positive* and *negative* training samples (correct algorithm, thrice slower)

## Triplet construction

The folder `data` contains a package for generating *triplets* to feed to a network.

To test the data script, run `data-test.lua`, but you need to have a dataset in the format described in `data.lua`.
In this same file is provided a snippet from the training script.
