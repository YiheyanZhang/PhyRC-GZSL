# Data layout

Download datasets from their official distribution pages and place them under `raw/`. The repository contains only the small class-level attribute JSON files in `processed/`.

Expected paths include:

```text
raw/
├── Pavia_University/PaviaU.mat
├── Pavia_University/PaviaU_gt.mat
├── Houston/Houston.mat
├── Houston/Houston_gt.mat
├── Indian_Pines/Indian_pines_corrected.mat
├── Indian_Pines/Indian_pines_gt.mat
├── WHU-Hi-LongKou/WHU_Hi_LongKou.mat
└── WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat
```

Additional dataset paths are declared directly in `../configs/*.yaml`.
