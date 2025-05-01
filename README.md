# OFI_Features

This repository constructs four Order Flow Imbalance (OFI) features per minute:

1. **Best-Level OFI** (level 0)  
2. **Multi-Level OFI** (levels 0–9)  
3. **Integrated OFI** (PCA compression of multi-level OFI)  
4. **Cross-Asset OFI** (sum of other symbols’ integrated OFI)

## Files

- `ofi_features.py`: Core module
- `notebooks/OFI_Feature_Construction.ipynb`: Demonstration notebook

## Requirements

```bash
pip install pandas numpy scikit-learn
