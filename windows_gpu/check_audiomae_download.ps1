$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -c "import timm; model=timm.create_model('hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m', pretrained=True, num_classes=4); print('AudioMAE loaded:', model.__class__.__name__)"
