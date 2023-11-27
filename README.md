<div align="center">
  
# DreamCreature: Crafting Photorealistic Virtual Creatures from Imagination

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://dreamwireart.github.io/"><img alt="page" src="https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white"></a>
[![Page Views Count](https://badges.toozhao.com/badges/01HG2ZDZV8WJ73GSR6PXBXAZ56/blue.svg)](https://badges.toozhao.com/badges/01HG2ZDZV8WJ73GSR6PXBXAZ56 "Get your own page views count badge on badges.toozhao.com")

</div>

![overview](docs/fig1.png)

**Abstract**: Recent text-to-image (T2I) generative models allow for high-quality synthesis following either text instructions or visual examples. Despite their capabilities, these models face limitations in creating new, detailed creatures within specific categories (e.g., virtual dog or bird species), which are valuable in digital asset creation and biodiversity analysis. 
To bridge this gap, we introduce a novel task, **Virtual Creatures Generation**: Given a set of unlabeled images of the target concepts (e.g., 200 bird species), we aim to train a T2I model capable of creating new, hybrid concepts within diverse backgrounds and contexts.
We propose a new method called **DreamCreature**, which identifies and extracts the underlying sub-concepts (e.g., body parts of a specific species) in an unsupervised manner. The T2I thus adapts to generate novel concepts (e.g., new bird species) with faithful structures and photorealistic appearance by seamlessly and flexibly composing learned sub-concepts. To enhance sub-concept fidelity and disentanglement, we extend the textual inversion technique by incorporating an additional projector and tailored attention loss regularization. Extensive experiments on two fine-grained image benchmarks demonstrate the superiority of DreamCreature over prior art alternatives in both qualitative and quantitative evaluation. Ultimately, the learned sub-concepts facilitate diverse creative applications, including innovative consumer product designs and nuanced property modifications.



![sourceAB](docs/fig2.png)

![composite](docs/fig3.png)

### Notes
We are working on releasing the code... 🏗️ 🚧 🔨 Please stay tuned!
