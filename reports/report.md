# Dataset requirements

* Minimum of a few hundred distinct images of plants across different ages
* Annotated with true leaf count
* Standardized image dimensions (256 x 256)
* Data augmentation to increase dataset size/variety
* Rotation
* Mirroring

# Approach

* CNN architecture with single continuous output
* Normalize inputs
* Try grayscale vs. color
* Round output to nearest natural number
* Similar research: https://arxiv.org/pdf/1708.07570.pdf
* OPO Implementation: https://bitbucket.org/onepointoneinc/leaf-count/src/master/

# Project Reflection:

After manually checking the labels provided by Mechanical Turk participants against their corresponding plant images, it was found that the labels were more variable than ideal, with different persons ostensibly using fairly different methods of determining the minimum and maximum number of leaves on a plant in a given image. This is related to an inherent difficulty to distinguish one leaf from another, especially since only one image perspective is provided. However, this was to be expected, hence the requirement of a range of possible leaf number to be provided by each participant. Moreover, this problem can be alleviated in the future by (1) being much more specific in the method of determination and (2) limiting participants to those who are considered reliable Mechanical Turk users. 

Images were augmented only through rotation, but another pass could be made also incorporating mirroring of images to further increase dataset size by a factor of two.

The loss function was defined as the absolute difference between the model's output and the average of the minimum and maximum.

The accuracy metric was defined as the number of predictions that were in the actual range of leaf number. After 22 epochs of training with tuned hyperparameters, an accuracy of 52% was achieved. However, this number should not be treated as the true efficacy of the model, for the actual ranges provided by Mechanical Turk participants were found to be inaccurate a significant amount of the time during manual checking, with a roughly equal probability of the participant overestimating vs. underestimating. This fact likely accounts for the fact that the accuracy is not higher, because during manual checking of predictions vs. actual ranges against their corresponding plant images, it was found that very often when a prediction fell outside of its actual range, the checker found the predicted number of leaves to be more accurate than the actual range. This indicates that the model is likely more efficacious than is indicated by its achieved accuracy metric.

It is suggested that a survey of true leaf counts be performed on the same species of plant, in order that its distribution may be compared with that of (1) the Mechanical Turk-derived data and (2) the predictions output by the model. If the true leaf count distribution is found to be more similar to (2) than to (1), then more credit can be given to the model's predictions. 