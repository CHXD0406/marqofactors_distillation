# marqofactors_distillation
A lightweight multimodal model for extracting visible product factors from noisy e-commerce image-title inputs.

## Model Architecture
Frozen DINOv2 + Frozen Chinese RoBERTa
 
  → Stage1 Text Denoising Q-Former
  
  → Stage2 Image Grounding Q-Former
  
  → Autoregressive Factor Decoder


imput: demo.jpg  title "2026新款Burberry风衣早秋长款男士外套官方正版英国代购"

output: ["卡其色","格纹元素","双排扣","腰带","毛呢/羊绒感"]
