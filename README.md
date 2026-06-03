
# Experiment: backprop alternative on an easy inverse problem.

<img width="168" height="93" alt="Screenshot 2026-06-03 at 4 27 15 PM" src="https://github.com/user-attachments/assets/cc4ed827-7724-4a12-85d6-b7960a6e1ee1" />

```
uv run linattention_solve.py
uv run linattention_visualize.py
```


## Easy problem

1. stripped down [Linear Transformer](https://manifestai.com/blogposts/faster-after-all/) (drop the softplus)
2. teacher is initialized with small rotations (Pi/10)
3. student is initialized with identity matrices
4. Initialize LR using line search, apply adaptive LR tuning at each step

report -- report here

Observations:

- Alternative allows using much larger learning rate and converges faster

![](images/image3.png)

- altprop changes last layer much more than it changes the first layer while regular backprop does the opposite -- the first layer is changed much more than the last layer.

![](images/image1.png)

![](images/image2.png)

([colab](https://colab.research.google.com/drive/1t3YD6hQsBcTwnaVPKxgVoMRo-idLjze3#scrollTo=rg9J-sPFR3Gd)).
