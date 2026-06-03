# Backprop vs altprop on an easy inverse problem.

```
uv run linattention_solve.py
uv run linattention_visualize.py
```

Altprop: use linear approximation to take into account the update to layer n+1 when computing update to layer n  ([colab](https://colab.research.google.com/drive/1t3YD6hQsBcTwnaVPKxgVoMRo-idLjze3#scrollTo=rg9J-sPFR3Gd)).

Easy problem

1. stripped down linear Transformer (drop the softplus)
2. teacher is initialized with small rotations (pi/10)
3. student is initialized with identity matrix
4. Use learning rate search for starting LR, and adaptive LR tuning at each step

Observations:

altprop allows using much larger learning rate and converges faster

![](images/image3.png)

altprop changes last layer much more than it changes the first layer while regular backprop does the opposite -- the first layer is changed much more than the last layer.

![](images/image1.png)

![](images/image2.png)

```
uv run linattention_solve.py
```

A bit more experiments with modified attention:
1. Learning rate is important. updating backprop lets you use much larger learning rate
2. The overall dynamics is different, Regular backprop biases more learning to happen in the first layer, while this backprop biases more learning to happen in the last layer
