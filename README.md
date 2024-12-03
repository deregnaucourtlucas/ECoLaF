# ECoLaF

This package offers a user-friendly toolkit to build an adaptive multimodal late fusion pipeline (*ECoLaF*) based on the Dempster-Shafer Theory from multiple single-modality neural networks.
The proposed pipeline can be used for semantic segmentation and classification tasks.

<!-- Mettre les équations et des exemples plus visuels pourrait pimper la description !!! -->

## INSTALLATION
To install the package, use the following command: `pip install ecolaf`.

## USAGE
Firstly, a single-modality neural network needs to be defined for each modality. Please make sure that each neural network has a *forward* method. If the dataset you are working with has *K* classes, please make sure that your models output *K+1* values to fit the Dempster-Shafer framework. 

Then you can build an *ECoLaF* pipeline as follows:   
`MyModel = ECOLAF(list_of_single_modality_networks, num_classes)`.

Given a list of multimodal images, the inference can be done as follows:   
`output = MyModel(list_of_images, **kwargs)`.

The optional arguments of the forward method are listed bellow:

```
output_unimodal (Boolean): returns the list of outputs for each model

output_conflict (Boolean): returns a conflict map of shape BxMxHxW

output_discounting_coef (Boolean): returns the discounting coefficients map of shape BxMxHxW

interpolation (Boolean): interpolates the output to the original image size

output_keyword (String): key to access to the output tensor if the model's output is a dictionary 
(for example, the deeplabV3_resnet50 model from torchvision requires output_keyword='out')

B: batch_size, M: number of modalities, H: images height, W: images width
```

An example is provided in `test/test_code.py`.

## CITATIONS

If you use ECoLaF, please cite the following work:

```
@InProceedings{deregnaucourt_2025_WACV,
    author    = {Deregnaucourt, Lucas and Laghmara, Hind and Lechervy, Alexis and Ainouz, Samia},
    title     = {A Conflict-guided Evidential Fusion for Multimodal Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
}
```
