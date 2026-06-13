GAN on MNIST : Handwritten Digit Generation
A Generative Adversarial Network (GAN) built from scratch in PyTorch, trained on the MNIST dataset to generate realistic handwritten digit images.
________________


Overview
This project implements a vanilla GAN with a deep, BatchNorm-regularized Generator and an MLP Discriminator. The two networks are trained adversarially: the Generator learns to produce images that fool the Discriminator, while the Discriminator learns to distinguish real MNIST digits from generated fakes.
Training was run for 200 epochs on a GPU (Google Colab T4), with sample grids saved at regular intervals to track visual progress.
________________


Architecture
Generator
Takes a random latent vector z (dim=100) and maps it to a 28×28 grayscale image.
z (100) → Linear(128) → BN → LeakyReLU
        → Linear(256) → BN → LeakyReLU
        → Linear(512) → BN → LeakyReLU
        → Linear(1024) → BN → LeakyReLU
        → Linear(784) → Tanh
        → reshape (1, 28, 28)


Discriminator
Takes a flattened 784-dim image and outputs a real/fake probability.
x (784) → Linear(512) → LeakyReLU
        → Linear(256) → LeakyReLU
        → Linear(1)   → Sigmoid


________________


Training Details
Hyperparameter
	Value
	Epochs
	200
	Batch size
	64
	Latent dimension
	100
	Learning rate
	0.0002
	Optimizer
	Adam (β1=0.5, β2=0.999)
	Loss function
	Binary Cross-Entropy
	Hardware
	NVIDIA T4 (Google Colab)
	

D loss formula: (real_loss + fake_loss) / 2
Sample interval: every 400 batches
________________


Results
By the end of training, the losses stabilise around:
* D loss ≈ 0.30
* G loss ≈ 1.7
This reflects a healthy adversarial equilibrium — the Discriminator is no longer trivially separating real from fake, and the Generator is producing credible digit images.
________________


Setup & Usage
Requirements
pip install torch torchvision numpy matplotlib


Run
Open GAN_MNIST_mine.ipynb in Google Colab (GPU runtime recommended) and run all cells. The MNIST dataset is downloaded automatically.
Generated image grids are saved to your Google Drive at /MyDrive/gan_images_mine/ at every 400-batch interval.
________________


Project Structure
├── GAN_MNIST_mine.ipynb   # Main notebook — full implementation & training
└── README.md


________________


Key Design Choices
* BatchNorm in the Generator — stabilises training and improves gradient flow across the deeper architecture, following the original DCGAN paper's recommendations for MLP GANs.
* LeakyReLU throughout — prevents dead neurons in both networks; slope = 0.2.
* Averaged Discriminator loss — dividing (real + fake) / 2 gives the Discriminator a loss on the same scale as the Generator, preventing it from dominating early training.
* No BatchNorm on the first Generator block — applied to the latent input directly, as normalising z can suppress diversity in generated outputs.
________________


References
* Goodfellow et al., Generative Adversarial Networks (2014)
* Radford et al., Unsupervised Representation Learning with Deep Convolutional GANs (2015)
* PyTorch Documentation
* MNIST Dataset