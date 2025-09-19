# Normalizing Image Tensors in PyTorch

When you normalize the tensors of images, you need to use the **mean** and **standard deviation (std)** of the dataset.  
This is commonly done with:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## Choosing Mean and Std

- Using a pretrained model?
    - Use the exact mean/std values that the model was trained with (e.g., ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).

- Training from scratch on your own dataset?
    - Compute the mean and std directly from your dataset (per channel).

- Just experimenting / toy project?
    - A simple (0.5, 0.5) works fine, which maps input values from [0,1] to [-1,1].



# There’s a concept called gradient accumulation:

- If your GPU can’t handle a big batch size, you can simulate it.
- Example: instead of 128 in one go, you do 4 mini-batches of 32, then update weights once → equivalent to batch size 128.


# Deep learning involves randomness in several places:

1. Weight initialization
- When you create a neural network (nn.Linear, etc.), the weights start with random values.
- Different starting points can lead to slightly different training results.

2. Data shuffling
- The DataLoader shuffles batches randomly each epoch.
- Different order → slightly different learning paths.

3. Dropout layers
- Dropout randomly turns off some neurons during training.
- Different random masks → slightly different gradient updates.

# Key techniques when designing image classifiers
### (a) Architecture choice
- MLP → simple tasks, small datasets (like MNIST).
- CNN (Conv2D + Pooling) → standard for most image classification.
- ResNet, EfficientNet, Vision Transformers → modern, scalable, state-of-the-art.

### (b) Number of layers & neurons
- More layers → higher capacity (can learn more complex patterns).
- But too many → risk of overfitting (memorizing training set).
- Rule of thumb: start small, then scale up if accuracy is low.

### (c) Activation functions
- ReLU is most common.
- Others: LeakyReLU, GELU (used in Transformers).

### (d) Regularization techniques
- Dropout (like in your code): randomly zeroes some neurons.
- Weight decay (L2 regularization): penalizes large weights.
- Data augmentation: rotate, flip, crop, noise in images → makes model robust.

### (e) Optimization
- Use optimizers: SGD, Adam (most common), AdamW.
- Learning rate is the most important hyperparameter.
- Use learning rate schedulers (reduce LR over time).