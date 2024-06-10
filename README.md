* This project makes use of images synthesized using openmoji dataset (https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip) to do object localization.
* The images are synthesized using a data generator and these are used for both training and testing.
* It is assumed that each each image has only one object and the synthesized image consisits of one emoji randomly placed in a white canvas. 
* The ground truth bounding box is represented by a green box and the predicted bounding box is represented by a red box. 
* If the predictions match the ground truth labels, the label colour in the output wil be green, else red.