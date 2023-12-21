<div align="center">
  
# DreamCreature: Crafting Photorealistic Virtual Creatures from Imagination

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Paper](http://img.shields.io/badge/Paper-arxiv.2311.15477-B31B1B.svg)](https://arxiv.org/abs/2311.15477)
[![Page Views Count](https://badges.toozhao.com/badges/01HG2ZDZV8WJ73GSR6PXBXAZ56/blue.svg)](https://badges.toozhao.com/badges/01HG2ZDZV8WJ73GSR6PXBXAZ56 "Get your own page views count badge on badges.toozhao.com")

</div>

![overview](docs/assets/fig1.png)

**Abstract**: Recent text-to-image (T2I) generative models allow for high-quality synthesis following either text
instructions or visual examples. Despite their capabilities, these models face limitations in creating new, detailed
creatures within specific categories (e.g., virtual dog or bird species), which are valuable in digital asset creation
and biodiversity analysis.
To bridge this gap, we introduce a novel task, **Virtual Creatures Generation**: Given a set of unlabeled images of the
target concepts (e.g., 200 bird species), we aim to train a T2I model capable of creating new, hybrid concepts within
diverse backgrounds and contexts.
We propose a new method called **DreamCreature**, which identifies and extracts the underlying sub-concepts (e.g., body
parts of a specific species) in an unsupervised manner. The T2I thus adapts to generate novel concepts (e.g., new bird
species) with faithful structures and photorealistic appearance by seamlessly and flexibly composing learned
sub-concepts. To enhance sub-concept fidelity and disentanglement, we extend the textual inversion technique by
incorporating an additional projector and tailored attention loss regularization. Extensive experiments on two
fine-grained image benchmarks demonstrate the superiority of DreamCreature over prior art alternatives in both
qualitative and quantitative evaluation. Ultimately, the learned sub-concepts facilitate diverse creative applications,
including innovative consumer product designs and nuanced property modifications.

### Notes

Code available now!! You can run this
at [colab](https://colab.research.google.com/drive/1gF6xIsC7ofM0zxoHl9zSPFiFXi-olmI0?usp=sharing) now!!

Please go to "https://xxx.gradio.live" after the public URL appears.

[//]: # (~We are working on releasing the code... 🏗️ 🚧 🔨 Please stay tuned!  &#40;I am cleaning up my messy code base & training a model with SDXL instead of SDv1.5 as mentioned in the paper&#41;)

### Methodology

![sourceAB](docs/assets/fig4.png)

Overview of our DreamCreature. (Left) Discovering sub-concepts within a semantic hierarchy involves partitioning each
image
into distinct parts and forming semantic clusters across unlabeled training data. (Right) These clusters are organized
into a dictionary,
and their semantic embeddings are learned through a textual inversion approach. For instance, a text description
like `a photo of a
[Head,42] [Wing,87]...` guides the optimization of the corresponding textual embedding by reconstructing the associated
image. To
promote disentanglement among learned concepts, we minimize a specially designed attention loss, denoted as
$\mathcal{L}_{attn}$.

### Mixing sub-concepts

![sourceAB](docs/assets/fig2.png)

Integrating a specific sub-concept (e.g., body, head, or even background) of a source concept B to the target concept A.

### Our results

Mixing 4 different species:

![composite](docs/assets/fig3.png)

More examples;

![more](docs/assets/moreexamples.png)

Creative generation:

![creative](docs/assets/creativegeneration.png)

### Todo

- train kmeans segmentation
- upload pretrained weights to huggingface
- requirements.txt
- update readme