<h1 align="center">Reproducability Study: Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery</h1>


### For an interactive walkthrough with our results, as well as some visualizations:
1. Clone this repository
2. Download <a href="https://drive.google.com/drive/folders/1Sy5qPX0tr2rLEI_6Sh5ZqTC-0kfKlLUL?usp=share_link">this</a> folder, and place the 2 folders inside this repository.
3. Open the notebook inside the repository on Google Colab


</p>

## Below, we have adapted the original README.md file from the authors of the paper, which can be found <a href="https://github.com/neuroexplicit-saar/Discover-then-Name/blob/main/README.md">here</a>.
</div>

## Setup

### Prerequisites

All the dependencies and packages can be installed using pip. The code was tested using Python 3.10.

### Installing the Packages

Use:

```bash
pip install -r requirements.txt
pip install -e sparse_autoencoder/
pip install -e .
```

### Dataset for training Sparse Autoencoder (CC3M)

#### Download the CC3M tar file to train the SAE
Note: Number of downloaded paired dataset might be less than used for training as the original authors downloaded the dataset in December, 2023. As of February 2025, about 68% of the links are still valid.

1) Download the ‘Train_GCC-training.tsv’ and ‘Validation_GCC-1.1.0-Validation.tsv’ from  https://ai.google.com/research/ConceptualCaptions/download by clicking on training split and validation split. 

2) Change their names to cc3m_training.tsv and cc3m_validation.tsv 

3) For training dataset: 
    ```bash
    sed -i '1s/^/caption\turl\n/' cc3m_training.tsv 
    img2dataset --url_list cc3m_training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder training --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
    ``` 

4) for validation dataset:
    ```bash
    sed -i '1s/^/caption\turl\n/' cc3m_validation.tsv 
    img2dataset --url_list cc3m_validation.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder validation --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
    ```

### Vocabulary for naming concepts

We use the vocabulary of 20k words used by [CLIP-Dissect](https://arxiv.org/abs/2204.10965), from [here](https://github.com/first20hours/google-10000-english/blob/master/20k.txt). Download and place the text file named as `"clipdissect_20k.txt` in `vocab_dir` specified in `config.py`. Then compute normalized CLIP embeddings of each text and save them as `embeddings_<encoder_name>_clipdissect_20k.pth` in `vocab_dir`. For example, for CLIP ResNet-50, the embedding file should be named `embeddings_clip_RN50_clipdissect_20k.pth`.

Additionally, add the path to the 'target_embeddings.pth' file 


### Datasets for training downstream probes

These are the datasets on which linear probes are trained on the learnt concept bottleneck to form a concept bottleneck model (CBM). In our paper, we use four datasets: Places365, ImageNet, CIFAR10, CIFAR100. Instructions for running experiments on these datasets is provided below, for other datasets you may need to define your own utils.

* Download the respective datasets:
    * [Places365](https://pytorch.org/vision/main/generated/torchvision.datasets.Places365.html)
    * [ImageNet](https://www.image-net.org/)
    * [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
    * [CIFAR100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
* Set the paths to the datasets in `config.py`.

NOTE: For Places365 we downloaded one in 10 files in from the original dataset due to computational constraints. This way roughly the same class distribution is preserved.


## Usage

The following shows example usage with CLIP ResNet-50 as the model, CC3M as the dataset for training the SAE, and Places365 as the dataset for downstream classification.

### Training a Sparse Autoencoder (SAE)


#### Save the CLIP features on CC3M to train the SAE on 

```bash
python scripts/save_cc3m_features.py --img_enc_name clip_RN50 
```

#### Train the SAE

```bash
python scripts/train_sae_img.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 200 --resample_freq 10 --ckpt_freq 0 --val_freq 1 --train_sae_bs 4096
```

### Assigning Names to Concepts

```bash
python scripts/assign_names.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 200 --resample_freq 10 --train_sae_bs 4096
```
 
### Training a Linear Probe for the Concept Bottleneck Model

#### Save the CLIP features of probe dataset

```bash
python scripts/save_probe_features.py --img_enc_name clip_RN50  --probe_dataset places365
```
#### (OPTIONAL: Cosine similarity finetuning for increased explainability)
As our extension to the original paper, we introduce a finetuning step, which results in more explainable concepts at the cost of a small accuracy decrease. This step results in learned concepts that are more similar to the true CLIP embeddings of that concept. For this finetuning step, we introduce the 'cosine coefficient', which controls the degree in which the learned embeddings are 'pushed' to the true CLIP embeddings. A cosine coefficient of 1.0 results in an average cosine similarity of ~1. We empirically found 1e-4 to be the optimal value in our experiments, leading to an average cosine similarity of ~0.54 after 30 epochs. 

```bash
python scripts/finetune_sae.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 30 --resample_freq 10 --ckpt_freq 0 --val_freq 1 --train_sae_bs 4096 --cosine_coefficient 1e-4
```

#### Save concept strengths using the trained SAE

Without finetuning:
```bash
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 0.0 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split train --num_epochs 200
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 0.0 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split train_val --num_epochs 200
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 0.0 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split val --num_epochs 200
```

With finetuning:
```bash
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 1e-4 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split train --num_epochs 30
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 1e-4 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split train_val --num_epochs 30
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 1e-4 --img_enc_name clip_RN50 --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split val --num_epochs 30
```

#### Train the probe on the saved concept strengths

Without finetuning:
```bash
python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 0.0 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 0 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset places365
```

With finetuning:
```bash
python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --cosine_coefficient 1e-4 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 30 --ckpt_freq 0 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset places365
```

### Visualization plots 
Example on the SAE's trained above (with and without finetuning)

#### Task Agnosticity

```bash
python scripts/visualization/vis_task_agnosticity_dump.py --img_enc_name clip_RN50 --probe_split val --method_name ours  --device cuda --sae_dataset cc3m --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --num_epochs 200 --resample_freq 10 --cosine_coefficient 0.0 --ckpt_freq 0 --val_freq 1 --train_sae_bs 4096

python scripts/visualization/vis_task_agnosticity_plot.py --img_enc_name clip_RN50 --probe_split val --method_name ours  --device cuda --sae_dataset cc3m --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --num_epochs 200 --resample_freq 10 --cosine_coefficient 0.0 --ckpt_freq 0 --val_freq 1 --train_sae_bs 4096
```

#### Local Explanation
Without finetuning:
```bash
python scripts/visualization/vis_local_explanations_dump.py --sae_dataset cc3m --img_enc_name clip_RN50 --method_name ours  --probe_split val --probe_dataset places365 --which_ckpt final --lr 0.0005 --l1_coeff 3e-5 --expansion_factor 8 --resample_freq 10  --num_epochs 200 --cosine_coefficient 0.0 --probe_lr 1e-2  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50

python scripts/visualization/vis_local_explanations_plot.py --sae_dataset cc3m --img_enc_name clip_RN50 --method_name ours --probe_split val --probe_dataset places365 --which_ckpt final --lr 0.0005 --l1_coeff 3e-5 --expansion_factor 8 --resample_freq 10  --num_epochs 200 --cosine_coefficient 0.0 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50
```

With finetuning:
```bash
python scripts/visualization/vis_local_explanations_dump.py --sae_dataset cc3m --img_enc_name clip_RN50 --method_name ours  --probe_split val --probe_dataset places365 --which_ckpt final --lr 0.0005 --l1_coeff 3e-5 --expansion_factor 8 --resample_freq 10  --num_epochs 30 --cosine_coefficient 1e-4 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50

python scripts/visualization/vis_local_explanations_plot.py --sae_dataset cc3m --img_enc_name clip_RN50 --method_name ours --probe_split val --probe_dataset places365 --which_ckpt final --lr 0.0005 --l1_coeff 3e-5 --expansion_factor 8 --resample_freq 10  --num_epochs 30 --cosine_coefficient 1e-4 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1.0 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50
```

## Acknowledgements

This repository uses code from the following repositories:

* [openai/CLIP](https://github.com/openai/CLIP)
* [ai-safety-foundation/sparse_autoencoder](https://github.com/ai-safety-foundation/sparse_autoencoder/)

## Citation

Please cite as follows:

```tex
@inproceedings{Rao2024Discover,
    author    = {Rao, Sukrut and Mahajan, Sweta and B\"ohle, Moritz and Schiele, Bernt},
    title     = {Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery},
    booktitle = {European Conference on Computer Vision},
    year      = {2024}
}
```
