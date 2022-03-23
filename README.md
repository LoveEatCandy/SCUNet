# Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis
![visitors](https://visitor-badge.glitch.me/badge?page_id=cszn/SCUNet) 

[[Paper]()][[Code]()]

Swin-Conv-UNet (SCUNet) denoising network
----------
<img src="figs/arch_scunet.png" width="900px"/> 

The architecture of the proposed Swin-Conv-UNet (SCUNet) denoising network. SCUNet exploits the swin-conv (SC) block as
the main building block of a UNet backbone. In each SC block, the input is first passed through a 1×1 convolution, and subsequently is
split evenly into two feature map groups, each of which is then fed into a swin transformer (SwinT) block and residual 3×3 convolutional
(RConv) block, respectively; after that, the outputs of SwinT block and RConv block are concatenated and then passed through a 1×1
convolution to produce the residual of the input. “SConv” and “TConv” denote 2×2 strided convolution with stride 2 and 2×2 transposed
convolution with stride 2, respectively.


New data synthesis pipeline for real image denoising
----------
<img src="figs/pipeline_scunet.png" width="900px"/> 

Schematic illustration of the proposed paired training patches synthesis pipeline. For a high quality image, a randomly shuffled
degradation sequence is performed to produce a noisy image. Meanwhile, the resizing and reverse-forward tone mapping are performed
to produce a corresponding clean image. A paired noisy/clean training patches are then cropped for training deep blind denoising model.
Note that, since Poisson noise is signal-dependent, the dashed arrow for “Poisson” means the clean image is used to generate the Poisson
noise. To tackle with the color shift issue, the dashed arrow for “Camera Sensor” means the reverse-forward tone mapping is performed on
the clean image.

<img src="figs/data_scunet.png" width="900px"/> 

Synthesized noisy/clean patch pairs via our proposed training data synthesis pipeline. The size of the high quality image patch is
544×544. The size of the noisy/clean patches is 128×128.

Results on Gaussian denoising
----------
<img src="figs/gray_scunet.png" width="900px"/>  

<img src="figs/comparison_scunet.png" width="900px"/>  


<img src="figs/color_scunet.png" width="900px"/>  


Results on real image denoising
----------
<img src="figs/real_scunet.png" width="900px"/>  


<img src="figs/real_scunet2.png" width="900px"/>  








