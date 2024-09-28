<div align="center">

# PartCraft: Crafting Creative Objects by Parts (ECCV 2024)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Paper](http://img.shields.io/badge/Paper-arxiv.2311.15477-B31B1B.svg)](https://arxiv.org/abs/2311.15477)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2407.04604-B31B1B.svg)](https://arxiv.org/abs/2407.04604)
[![Hugging Face](https://img.shields.io/badge/DreamCreature-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/kamwoh/dreamcreature)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamwoh/partcraft/blob/master/dreamcreature_gradio.ipynb)

</div>

![overview](docs/assets/newfig1.png)

**Abstract**: This paper propels creative control in generative visual AI by allowing users to "select". Departing from traditional text or sketch-based methods, we for the first time allow users to choose visual concepts by parts for their creative endeavors. The outcome is fine-grained generation that precisely captures selected visual concepts, ensuring a holistically faithful and plausible result. To achieve this, we first parse objects into parts through unsupervised feature clustering. Then, we encode parts into text tokens and introduce an entropy-based normalized attention loss that operates on them. This loss design enables our model to learn generic prior topology knowledge about object's part composition, and further generalize to novel part compositions to ensure the generation looks holistically faithful. Lastly, we employ a bottleneck encoder to project the part tokens. This not only enhances fidelity but also accelerates learning, by leveraging shared knowledge and facilitating information exchange among instances. Visual results in the paper and supplementary material showcase the compelling power of **PartCraft** in crafting highly customized, innovative creations, exemplified by the "charming" and creative birds.


### Methodology

![methodology](docs/assets/newfig4.png)

Overview of our PartCraft. (Left) Part discovery within a semantic hierarchy involves partitioning each
image into distinct parts and forming semantic clusters across unlabeled training data.
(Right) All parts are organized into a dictionary, and their semantic embeddings are learned through a textual inversion approach.
For instance, a text description like `a photo of a [Head,42] [Wing,87]...` guides the optimization of the corresponding textual embedding by reconstructing the associated image.
To improve generation fidelity, we incorporate a bottleneck encoder $f$ (MLP) to compute the embedding $y$ as input to the text encoder.
To promote disentanglement among learned parts, we minimize a specially designed attention loss, denoted as
$\mathcal{L}_{attn}$.

### Mixing sub-concepts

![sourceAB](docs/assets/fig2.png)

Integrating a specific part (e.g., body, head, or even background) of a source concept B to the target concept A.

### Our results

Mixing 4 different species:

![composite](docs/assets/fig3.png)

More examples;

![more](docs/assets/moreexamples.png)

Creative generation:

![creative](docs/assets/creativegeneration.png)

### Usage

1. A demo is available on
   the [`kamwoh/dreamcreature` Hugging Face Space](https://huggingface.co/spaces/kamwoh/dreamcreature). (Very very slow
   due to CPU only)
2. You can run the demo on a
   Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamwoh/dreamcreature/blob/master/dreamcreature_gradio.ipynb).
3. You can use the gradio demo locally by running `python app.py` or `gradio_demo_cub200.py` or `gradio_demo_dog.py` in
   the `src` folder.

### Training

1. Check out `train_kmeans_segmentation.ipynb` to obtain a DINO-based KMeans Segmentation that can obtain the "parts"/"
   sub-concepts". This is to obtain the "attention mask" used during the training.
2. Assuming no labels, we can also use the kmeans labels as a supervision, otherwise we can use the supervised labels (
   such as ground-truth class) as we can obtain higher quality of reconstruction.
3. Check out `run_sd_sup.sh` or `run_sd_unsup.sh` for training. All hyperparameters in these scripts are used in the
   paper.
4. SDXL version also available (see `run_sdxl_sup.sh`) but due to resource limitation, we cannot efficiently train a
   model, hence we do not have a pre-trained model on SDXL.

### Notes

1. The original paper title was: `DreamCreature: Crafting Photorealistic Virtual Creatures from Imagination`

### Todo

- [ ] Pre-trained model on unsupervised KMeans Labels as we used in the paper (CUB200)
- [ ] Pre-trained model on unsupervised KMeans Labels as we used in the paper (Stanford Dogs)
- [ ] Evaluation script (EMR & CoSim)
- [ ] Update readme
- [ ] Update website

### Citation

```
@inproceedings{
  ng2024partcraft,
  title={PartCraft: Crafting Creative Objects by Parts},
  author={Kam Woh Ng and Xiatian Zhu and Yi-Zhe Song and Tao Xiang},
  booktitle=ECCV,
  year={2024}
}
```
