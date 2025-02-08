# NoPrompt-UAD: An Approach for Prompt-less Unsupervised Anomaly Detection in Brain MRI Scans

This repository contains the code implementation for the paper "NoPrompt-UAD: An Approach for Prompt-less Unsupervised Anomaly Detection in Brain MRI Scans" by Bheeshm Sharma, Karthikeyan Jaganathan, and P. Balamurugan.

## Table of Contents
- [Abstract](#Abstract)
- [Environment Set-up](#environment-set-up)
- [DataSets](#datasets)
- [Running NoPrompt-UAD](#Running-NoPrompt-UAD)
  
## Abstract
Unsupervised anomaly detection (UAD) in brain MRI scans is an important challenge useful to obtain quick and accurate detection of brain anomalies when precise pixel-level anomaly annotations are unavailable. Existing UAD methods including diffusion models like DDPM and its variants pDDPM, mDDPM, cDDPM, and MCDDPM often suffer from prolonged training times. Prompt-based approaches like MedSAM can be used for UAD for inference with user-driven prompts, however, such models are trained on massive datasets with supervision. Further, most of these models are memory-heavy. In this work, we introduce NoPrompt-UAD, a novel UAD approach which eliminates the need for any user-driven prompts and pixel-level anomaly annotations. Our approach starts with a set of fixed candidate location prompts which are then enriched using an attention mechanism guided by image features to result in region-aware spatial point embeddings. These embeddings are then used in an anomaly mask decoder along with the image embeddings to obtain pixel-level anomaly annotations.

![NoPrompt-UAD Overview](/Figures/NoPrompt-UAD-Figure.png)
 

## Environment Set-up
To set up the environment, use the following installation instructions.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/bheeshmsharma/NoPrompt-UAD.git
    
3. Navigate to the project directory:
    ```bash
    cd NoPrompt-UAD
    ```
4. Create and activate the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate NoPrompt-UAD
    ```

## DataSets
This project utilizes the following datasets:
- **BraTS20**: [Brain Tumor Segmentation Challenge 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html).  
  - **Download Working Link**: [BraTS20 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?resource=download-directory)

- **BraTS21**: [Brain Tumor Segmentation Challenge 2021 dataset](http://braintumorsegmentation.org/).  
  - **Download Working Link**: [BraTS21 Dataset on Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)

- **BraTS23**: [Brain Tumor Segmentation Challenge 2023 dataset](https://www.synapse.org/Synapse:syn51156910/wiki/621282).  
  - **Download Working Link**: [BraTS23 Dataset on Kaggle](https://www.kaggle.com/datasets/shakilrana/brats-2023-adult-glioma)

- **MSD**: [Medical Segmentation Decathlon dataset](http://medicaldecathlon.com/).  
  - **Download Working Link**: [MSD Dataset on google drive]([https://www.kaggle.com/datasets/shakilrana/brats-2023-adult-glioma](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2))

- **MSLUB**: [The Multiple Sclerosis Dataset from The University Hospital of Ljubljana](https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/).
  - **Download from the above link**

## Data Preprocessing
Follow these steps to prepare the data for processing:

1. **Download the Dataset**:
   - Download the dataset and save it in the `DATA` directory of the project.

2. **Organize the Dataset**:
   - Ensure that the volumes are saved in one folder and the corresponding GT (Ground Truth) masks are saved in a separate folder within the `DATA` directory.

3. **Set Up Paths**:
   - Open the `Data_Extraction.py` script in the `DATA` directory.
   - Modify the paths in the script to point to the correct locations of your downloaded dataset and masks.

4. **Run the Data Extraction Script**:
   - After adjusting the paths, run the `Data_Extraction.py` script to process the data.

5. **Check Folder Structure**:
   - Once the extraction is complete, make sure the extracted data is saved in the suggested folder structure outlined in the `DDPT/readme.md`.

   > **Note**: Ensure that the final structure of the data follows the guidelines provided in the `DDPT/readme.md` file for consistency.


### Dataset Details

The table below provides information about the dataset details in this project:
![Dataset Split Info](/Figures/Data_split_info.png)

## Running NoPrompt-UAD

1. **Environment Setup**:  
   - Make sure you have followed the environment setup instructions to configure your dependencies and environment properly.
     
1. **Candidate Location Embedding Setup**:  
   - To save the Candidate Location Embedding, navigate to the `Fixed_Candidate_Embeddings` directory.
   - Once inside the directory, run the following `ipynb` file:
     ```
     Fixed_Candidate_Location_Embedding.ipynb
     ```
    - This will automatically save the Fixed Candidate Location Prompt Embedding within the `Fixed_Candidate_Embeddings` directory.

2. **Training with NoPrompt-UAD**:  
   - To train the model, use the following command:  
     ```bash
     python run.py
     ```  
   - Inside `run.py`, set all the training-related arguments, such as:  
     - Dataset  
     - Number of epochs  
     - Learning rate  
     - Early stopping criteria  
     - And other hyperparameters  

3. **Testing the Model**:  
   - To test the trained model, use the following command:  
     ```bash
     python test_run.py
     ```  
   - Similar to `run.py`, set the testing-related arguments in `test_run.py`, such as:  
     - Dataset  
     - Model checkpoint path  
     - Other Inference settings  

<!--
### Qualitative results:
We present below a few comparisons in terms of qualitative and quantitative results.
<img alt="image" src="images/Qualitative_Results.png" style="width: 100%;" height=500>
### Quantitative results:
![Dataset Split Info](/Figures/Data_split_info.png)

## Citation
If you use this code in your research, please cite our paper:

This project draws inspiration and is developed based on the [pddpm-uad](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD) repository.
-->
