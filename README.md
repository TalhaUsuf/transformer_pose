

# Positional Encoding Understanding 🕸

![](img.png)

[this](https://github.com/TalhaUsuf/multidim-positional-encoding.git) awesome 🔥 package has been
used for using the positional encodings.
install it using:

```bash
pip install positional-encodings[pytorch,tensorflow]
```



# Transformer model 🛠
for understanding the pytorch transformer masks, see below link
```html
https://discuss.pytorch.org/t/understanding-mask-size-in-transformer-example/147655/2
```
transformer model uses **2** types of masks 🚀
 - zero padding mask
 - look ahead mask
the link above will help you understand the masks in transformer model

> as for this model neither of the masks are used, but you can use them if sequence lengths of videos are different
> this implementation also only uses the **encoder** part of the transformer model since sequence based decoder is not needed for this task
