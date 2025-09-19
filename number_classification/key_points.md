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