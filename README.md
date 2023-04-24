# image_processing_final_project

## add new args: noise, prob
noise:jitter,flip,label,gaussian,gray,none
prob:between 0 and 1
gaussian std mean changes need to modify gaussian blur file in the SimCLR/data_aug/gaussian_blur.py
run python resnet18_supervised_loss.py --download --noise none --prob 0
to start supervised loss training without noise.
