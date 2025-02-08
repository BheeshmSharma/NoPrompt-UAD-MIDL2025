## MedSAM inference using DDPT mask guided Prompt (Box, Point)
1. In `MedSAM_Inference_DDPT-Guided.py`, update the directory path and folder name to match the dataset structure within the `/DATA/Dataset_name/` directory.
2. Adjust the prompt type to either `Box` or `Point`.
3. Execute the following command to save the masks, guided by DDPT masks:
   ```bash
   python MedSAM_Inference_DDPT-Guided.py
   ```
