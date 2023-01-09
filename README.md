# GaussLabs Mini Project
## Introduction
- Paper : Reversible Instance Normalization for accurate time series forecasting against distribution shift, Taesung kim et al
- link : https://openreview.net/pdf?id=cGDAkQo1C0p 
### Purpose
- Review the paper : RevIN, Reversible Instance Normalization method for data distribution shift issue in time-series
- Experiment for checking the validity of the RevIN
- Analysis for effect of RevIN with respect to the data distribution shift

### Reversible Instance Normalization
<div>
    <img src="/Image/PICTURE_01.png"  width="640" height="196">
</div>
- simple, flexible and general trainable layer for any arbitrarily chosen layers
- easy to use for any model architecture
- The example of usage is here

```
### Example code for usage of RevIN
from src.RevIN import RevIN
revin = RevIN(num_features)
inputs_norm = revin(inputs, 'norm') # Normalization process
outputs = model(inputs_norm) # Forward propagation
output_denorm = revin(outputs, 'denorm') # Denormalization process
```

- The code explaination is here
<div>
    <img src="/Image/PICTURE_02.png"  width="480" height="256">
</div>

## Result for experiments 
### Effect of RevIN on data distribution shift
<div>
    <img src="/Image/PICTURE_03.png"  width="480" height="256">
</div>

### Model performance enhancement from normalization : Comparison with other method
<div>
    <p float = 'left'>
        <img src="/Image/PICTURE_04.png"  width="480" height="256">
    </p>
</div>