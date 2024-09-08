# WinCLIP (Zero-Shot)
Pytorch implementation of WinCLIP paper for CVPR 2023: [WinCLIP](https://arxiv.org/pdf/2303.14814).  
Most codes were obtained from the following GitHub page: [[Link]](https://github.com/zqhang/Accurate-WinCLIP-pytorch)  

### The network pipeline.  
![pipe](https://github.com/user-attachments/assets/2ec04cf1-1160-4dcc-85e1-97e2438a4e87)

## Environments  
PyTorch >= 1.1.  
Python >= 3.6.  
sklearn  
opencv  
torchvision  
Other common packages.  

## Dataset
- **MVTec AD** dataset is a benchmark collection for anomaly detection in industrial images, featuring various objects and textures with annotated defects.
- Download ```meta.json``` and move it to the downloaded ```mvtec``` dataset directory.

|     MVTec AD            |     meta.json            |
|:------------------------:|:------------------------:|
| [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)   | [Google Drive](https://drive.google.com/file/d/11AdQpF3bhVCI0PoIVkku2OwJpoKKiJC_/view?usp=drive_link)  

## Zero-Shot AD
- When program starts, the ```results``` directory is automatically created.
- The result of anomaly detection (pickle file) is automatically saved in the ```results/{dataset}``` directory.
  
```Shell
# command example 
python run.py --dataset 'mvtec' --data_path '/home/sha/datasets/mvtec' \
   --model 'ViT-B-16-plus-240' --pretrained 'openai' --image_size 240 \
   --batch_size 64
```

## Visualization
- You can use the pickle file (e.g. mvtec_results.pickle) saved in the ```results/{dataset}``` directory for visualization.
- You can choose between two modes: ```heat``` and ```attn```. To use the attn mode, set ```attn_mode``` to True. 
```Shell
# command example 
python run.py --dataset mvtec --load_results 'mvtec_results' --image_size 240 --vis True --attn_mode False
```

## Results
| objects    |   AUC (Pixel-level) |       AUC (Image-level) | 
|:-----------|-----------:|-----------:|
| carpet     |       90.9  |       99.3 |    
| bottle     |       85.7 |         98.6 |    
| hazelnut   |       95.7 |          92.3 |   
| leather    |       95.5 |         100   |  
| cable      |       61.3 |         85   |   
| capsule    |       87   |          68.7 |    
| grid       |       79.4 |          99.2 |   
| pill       |       72.7 |          81.5 |   
| transistor |       83.7 |        89.1 |    
| metal_nut  |       49.3 |          96.2 |   
| screw      |       91.1 |         71.7 |    
| toothbrush |       86.2 |          85.3 |   
| zipper     |       91.7 |         91.2 |    
| tile       |       79.1 |          99.9 |   
| wood       |       85.1 |           97.6 |  
| mean       |       82.3 |           90.4 |    
#### Validation results can be found on the path ```results/{dataset}/imgs/{mode}```.  
| Mode: heat (heatmap)                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![seg](https://github.com/user-attachments/assets/b8aaed5a-0f1b-49d0-b302-09317b77d098)

| Mode: attn (attention)                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![attn](https://github.com/user-attachments/assets/cdfa022a-a8e6-4004-af03-d60ff88a1e4f)


