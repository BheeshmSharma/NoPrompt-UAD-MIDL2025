# DDPT-Discriminative Dual Prompt Tuning for Weakly Supervised Brain Anamoly Segmentation

<!--
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
-->

This work focuses on developing a weakly supervised anomaly detection approach for brain MRI scans, leveraging large-scale pre-trained vision-language models like CLIP. The project introduces a novel method using the Dual Prompt Tuning (DPT) paradigm to enhance both the classification and segmentation of anomalous regions in MRI images. This technique addresses the challenge of limited annotated medical data by effectively identifying regions of interest without relying on pixel-level annotations. The model integrates textual and visual cues through advanced tuning mechanisms, such as Contextualized Attentional Vision Prompt Tuning (CAVPT), to improve anomaly localization. This approach demonstrates significant potential in real-time applications, offering a scalable solution for medical image analysis while providing insights into cross-domain generalizability and optimization of medical methodologies.


![DDPT-Architecture](./DDPT_Architecture.png)


# Background

The background of this work lies in the critical need for efficient and accurate anomaly detection in medical imaging, particularly brain MRI scans. Traditional segmentation methods, which are essential for identifying regions of interest for medical interventions, often rely on extensive pixel-level annotations. This manual process is time-intensive, requires high domain expertise, and is not scalable in clinical environments. To address this challenge, weakly-supervised learning methods have gained traction, leveraging minimal supervision, such as image-level labels, to detect and localize anomalies. The advent of large-scale vision-language models like CLIP has further revolutionized the field by enabling effective representation learning from both textual and visual data. This study builds on these advances, introducing a novel discriminative network combined with the Dual Prompt Tuning (DPT) paradigm. By integrating concepts from prompt engineering and attention mechanisms, the work demonstrates a robust framework for transforming limited data into actionable insights, with applications extending to real-time segmentation tasks in diverse medical domains.



<!--

For a detailed introduction to SPG, please refer to **the [《SPG White Paper》](https://spg.openkg.cn/en-US "SPG White Paper") jointly released by Ant Group and OpenKG**.

-->

# How to use

This code is highly inspired by the official [DPT github](https://github.com/fanrena/DPT). The setup steps are similar to that repository and are shown below.

## Dataset Preparation

* Please ensure that the final structure of the data follows the suggested structure below: 

```
DATA/
|
└── <dataset_name>/
    |
    └── images/
        |
        ├── <Healthy>/
        |
        └── <Unhealthy>/
```

Note that the folder name inside of `images/` is also the class name that will be used later down. Inside these 2 folders upload the images. The splitting into test and train will be done by the system itself.

* Inside the datasets folder, create a `.py` file for this new dataset, which can use a reference from `caltech101.py`. **Make sure to set the class name specifically here**
* Update the `__init__.py` inside the datasets folder.
* In the `configs/datasets/` folder, create a `.yaml` file for the new dataset, and mention the configurations for the dataset.

## Updating run.sh

* In the `run.sh` file, update the path of the `yaml` file:
  ```bash
  --dataset-config-file configs/datasets/${DATASET}.yaml \ (line number 26)

* For all the other parameters you are interested in changing, make the changes in the `run.sh` file

* Run the following command to train and test the DDPT. 
    ```bash
    bash run.sh
* Now weak mask corresponding to the DDPT has been saved in the `DATA` directory.


Additional parameters to change in `run.sh`
* MODEL - Changes the model, and options - [DPT, VPT, VLP]
* DATASET - Enter the class name you created in the datasets folder
* DATADIR - Leave empty if the above instructions are to be followed
* EVAL - Set True to create maps, and run evaluation
* CLASSIFY - Set True to run the classification approach for the model with no information from the dataset
* TRAIN_CRAFT and EVAL_CRAFT - Set True based on where you want to introduce the hand-crafted prompts or learnable prompts (CoOp)
* EPOCHS - Set the value for training and loading the model for evaluation
* MODEL_DIR - Set the location of where to store the model weights
* NUM_SHOTS - Number of shots during training
* THRESHOLD - The samples having ground truth lesser than the set value of pixels will not be included in the evaluation step
* MEDIAN - Set True if you want to do Median filtering

## Modes of running code

1) Only train the model

    EVAL - False

2) Train the model and evaluate based on only true labels from the dataset

    EVAL - True

    CLASSIFIY - False

2) Train the model and evaluate based on the classification done by the model

    EVAL - True

    CLASSIFIY - True




<!--
# Cite

If you use this software, please cite it as below:
* [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation](https://arxiv.org/abs/2409.13731)
* KGFabric: A Scalable Knowledge Graph Warehouse for Enterprise Data Interconnection

```bibtex
@article{liang2024kag,
  title={KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation},
  author={Liang, Lei and Sun, Mengshu and Gui, Zhengke and Zhu, Zhongshu and Jiang, Zhouyu and Zhong, Ling and Qu, Yuan and Zhao, Peilong and Bo, Zhongpu and Yang, Jin and others},
  journal={arXiv preprint arXiv:2409.13731},
  year={2024}
}

@article{yikgfabric,
  title={KGFabric: A Scalable Knowledge Graph Warehouse for Enterprise Data Interconnection},
  author={Yi, Peng and Liang, Lei and Da Zhang, Yong Chen and Zhu, Jinye and Liu, Xiangyu and Tang, Kun and Chen, Jialin and Lin, Hao and Qiu, Leijie and Zhou, Jun}
}
```

# License

[Apache License 2.0](LICENSE)

-->
