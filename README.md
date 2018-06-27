# char-vae
Generating embedding vectors of characters from their images via VAE

## Making Vectors

Prepare a target font file.
For example, you can get a free font from
https://okoneya.jp/font/download.html


Make training images with the target font
```:bash
python gen_char_images.py <path and name of your font file>
```
char_image.pklb will be generated.


Run training script with the generated file.
```:bash
python charvae_train_v0.py
```
Note: Cofigurations for training are specified in charvae_train_v0.py
