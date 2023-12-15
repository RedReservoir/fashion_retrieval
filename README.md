# fashion_retrieval

## Module changes

  - `transformers`
    - `/models/cvt/modeling_cvt.py`
      - Added `contiguous()` after all `permute()`. This is done to fix the following error while using HuggingFace CVT model with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```
      Original code can be found in `/models/cvt/modeling_cvt__ORIGINAL.py`

  - `torchvision`
    - `/models/swin_transformer.py`
      - Added `contiguous()` after all `permute()`. This is done to fix the following error while using HuggingFace SwinT model with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```
      Original code can be found in `/models/swin_transformer__ORIGINAL.py`
    - `/models/convnext.py`
      - Added `contiguous()` after all `permute()` in class `LayerNorm2D`. This is done to fix the following error while using HuggingFace SwinT model with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```
      Original code can be found in `/models/convnext__ORIGINAL.py`
    - `/ops/misc.py`
      - Added `contiguous()` to `permute()` operation of `Permute` class. This is done to fix the following error while using models using this class with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```
      Original code can be found in `/ops/misc__ORIGINAL.py`

  - `fastervit`
    - `/models/faster_vit.py`
      - Added `contiguous()` after `permute()` in line 92. This is done to fix the following error while using model with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```

  - `src/external`
    - `/gc_vit/gc_vit.py`
      - Added `contiguous()` after `permute()` in lines 90, 101, 363, 437, 600. This is done to mitigate (not resolved) the following error while using model with PyTorch DDP:
      ```
      UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
      ```