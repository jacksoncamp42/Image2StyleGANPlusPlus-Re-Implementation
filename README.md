Introduction
In this project, we explore the advancements in Generative Adversarial Networks (GANs) for high-fidelity image editing by re-implementing the paper titled \textit{Image2StyleGAN++: How to Edit the Embedded Images?}. The authors, Rameen Abdal, Yipeng Qin, and Peter Wonka, expand upon their previous work, Image2StyleGAN, by introducing several significant enhancements. It was presented at CVPR 2020.

The core method discussed in the paper involves embedding images into the latent space of a pre-trained StyleGAN. This is achieved through an embedding algorithm that not only optimizes the latent space but also manipulates the noise and activation tensors to produce localized and high-fidelity edits on images.

The main contributions of the paper are threefold: Firstly, the introduction of noise optimization to enhance the reconstruction of high-frequency features in images, significantly improving the quality of the output. Secondly, the extension of the global W+ latent space embedding allows for local modifications and more controlled edits. Thirdly, the combination of latent space embedding with direct manipulation of activation tensors enables both global and local edits, providing a powerful framework for various image editing applications such as image inpainting, style transfer, and feature modification.

Chosen result

Identify the specific result you aimed to reproduce and its significance in the context of the paper’s main contribution(s).
• Include the relevant figure, table, or equation reference from the original paper.

We aimed to reproduce two specific results the first is the merging of two halves of an image as seen below. The significance of this result is that it is equally as good as copying and pasting the two halves of the image and producing a high-quality image with no splits. This could reduce the human work and shows that deep learning networks can be do just as much as manually cropping

The second which is making the PSNR ratio up from around 19 to 22 to 39 to 45 on recreated images. The significance is that we can recreate the image with a higher quality than before as it has a higher precision-to-noise ratio.

Reimplementation details
During this re-implementation, we started from the image2styleGAN paper and the pre-trained model, which we loaded via a pickle file. The main change in the model architecture was via the loss function and adding the noise optimization. When changing the loss function we changed it by adding multiple masks on the four different loss compared to the MSE loss and the perceptual loss in the original paper. In this loss function, we added two new parts and added masks primarily so that there could be style transfer from two images. We added the loss called the paper as the style loss which was taken from the third layer of the VGG16 pre-trained ImageNet model and the image not used in the style loss. The next thing we had to implement was masks which was a convolutional filter that specified which part of the image we took from image one and which part of the image one we took from image two. There had to be three separate filters because they were different parts of the loss and had different losses. However, overall they are supposed to roughly represent the same portion of our image. While doing this for the perceptual loss it was tough just because there were parts from different sizes so we couldn't use the same matrix. As a result we had a few matrices representing the same part of the image. These matrices would be hyperparameters to the loss functions as well as the images. In addition to blend two image, such as the image with a scribble result we would put in some type of blur matrix. Some of the shapes in the image were complicated to make as filters so we took the one that combined two images vertically.

For running our code we did it on a Jupiter notebook so one could simply run the Jupiter notebook. We used pytorch, numpy and pickle to load the pretrained styleGan network

We were able to reproduce results by running a V100 GPU for 20-30 minutes

Results and analysis

• Present your re-implementation results and compare them to the original paper’s findings.
