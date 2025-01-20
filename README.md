# Project Status
### 🔴 Closed

# Attention-U-Net-with-CRF

This project is a implementation of this paper: ["Attention U-Net: Learning Where to Look for the Pancreas"](https://arxiv.org/pdf/1804.03999), using a [ConditionalRandomField](https://en.wikipedia.org/wiki/Conditional_random_field) as a post-processing layer in order to **better segment small pieces of litter that have complex edges**. Here is an image \ mask example, which normal U-Nets do not handle well.


<p float="left">
  <img src="https://github.com/DanLaurentiu1/Attention-U-Net-with-CRF/blob/main/presentations/image_example.jpg" width="400" height="400"/>
  <img src="https://github.com/DanLaurentiu1/Attention-U-Net-with-CRF/blob/main/presentations/mask_example.png" width="400" height="400"/> 
</p>

**for more details, please go [here](https://github.com/DanLaurentiu1/Attention-U-Net-with-CRF/tree/main/presentations)**
