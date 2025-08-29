# ELF
===========
## Ensemble learning of foundation models for precision oncology

**Abstract:** Histopathology is essential for disease diagnosis and treatment decision-making. Recent advances in artificial intelligence (AI) have enabled the development of pathology foundation models that learn rich visual representations from large-scale whole-slide images (WSIs). However, existing models are often trained on disparate datasets using varying strategies, leading to inconsistent performance and limited generalizability. Here, we introduce ELF (Ensemble Learning of Foundation models), a novel framework that integrates five state-of-the-art pathology foundation models to generate unified slide-level representations. Trained on 53,699 WSIs spanning 20 anatomical sites, ELF leverages ensemble learning to capture complementary information from diverse models while maintaining high data efficiency. Unlike traditional tile-level models, ELF’s slide-level architecture is particularly advantageous in clinical contexts where data are limited, such as therapeutic response prediction. We evaluated ELF across a wide range of clinical applications, including disease classification, biomarker detection, and response prediction to major anticancer therapies—cytotoxic chemotherapy, targeted therapy, and immunotherapy—across multiple cancer types. ELF consistently outperformed all constituent foundation models and existing slide-level models, demonstrating superior accuracy and robustness. Our results highlight the power of ensemble learning for pathology foundation models and suggest ELF as a scalable and generalizable solution for advancing AI-assisted precision oncology.

## Dependencies:

**Hardware:**
- NVIDIA GPU (Pretrained on NVIDIA H100 x8 and tested on NVIDIA L40S x8) with CUDA 12.4 (Ubuntu server).

**Software:**
- Python (3.10.12), PyTorch (PyTorc2.6)

**Additional Packages/Libraries:**
* CLAM (https://github.com/mahmoodlab/CLAM)
* UNI (https://huggingface.co/MahmoodLab/UNI)
* CONCHV1.5 (https://huggingface.co/MahmoodLab/conchv1_5)
* Gigapath (https://huggingface.co/prov-gigapath/prov-gigapath)
* Virchow2 (https://huggingface.co/paige-ai/Virchow2)
* H-optimus-0 (https://huggingface.co/bioptimus/H-optimus-0)
* CHIEF (https://github.com/hms-dbmi/CHIEF)
* Prov-GigaPath (https://huggingface.co/prov-gigapath/prov-gigapath)
* TITAN (https://huggingface.co/MahmoodLab/TITAN)


## Step 1: Patch-level embedding extraction
* Extracting patch-level embedding using [CLAM](https://github.com/mahmoodlab/CLAM) and five tile-level foundation models ([UNI](https://huggingface.co/MahmoodLab/UNI), [CONCHV1.5](https://huggingface.co/MahmoodLab/conchv1_5), [Gigapath](https://huggingface.co/prov-gigapath/prov-gigapath), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)) in 10X (Extracted EBRAINS features for the tutorial are [here](https://drive.google.com/file/d/16tpUS-o21WsQH1U3Jyqi4784sb-OceiB/view?usp=sharing)).  

## Step 2: Slide encoder pretrain 
* Start training slide encoder using `sh bash_multiple.sh`, at least 8 x NVIDIA H100 needed. Logs and checkpoints will be saved to the default path.
* The pretrained model will be downloaded at [here](https://drive.google.com/file/d/1eotBSohYE9vy71a-LiNxI3ZP4reUN44L/view?usp=sharing).

## Step 3: Slide-level embedding extraction
* Patch-level embedding extraction following Step 1.
* Slide-level embedding extraction using `extract_multiple_model.sh` after setting the parameters successfully.

## Step 4: Evaluation on classification and regression tasks

* Evaluation on the downstream tasks, like the classification and regression examples in the `evaluation` folder. Please download and uncompress the slide embeddings of all comparison methods (Prov-GigaPath, CHIEF, TITAN and ELF) for evaluation, [evaluation](https://drive.google.com/drive/folders/1pvteAGR5y8UsTJ23VEPYrLSRRYpZsERh?usp=sharing). Note that,Prov-GigaPath, CHIEF, TITAN were extracted with the same WSI processing settings.
* Each tutorial can be run via `sh XXX.sh` (for ebrains, `sh ebrains.sh`), and the comparison results will be saved to their corresponding folder.



## Acknowledgments
This project builds upon many open-source repositories such as CLAM (https://github.com/mahmoodlab/CLAM), COBRA (https://github.com/KatherLab/COBRA), mocov3(https://github.com/facebookresearch/moco-v3), and TITAN (https://github.com/mahmoodlab/TITAN). We thank the authors and contributors to these repositories.
## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
## Citation
If you find our work useful in your research, please consider citing:



