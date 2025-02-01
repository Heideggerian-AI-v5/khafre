This example showcases functional object segmentation. Functional parts of objects such as handles, spikes, or blades are relevant to detect because that is where affordances of objects can be manifested. Specialized object detection models can be trained to detect such parts, but it is also interesting for a perception system to be able to collect its own training data in an autonomous way, based on the behaviors it observes, and the knowledge it has of how an affordance manifestation would look like. 

In this example we showcase the selection of parts of an object which are observed to behave in particular ways, specified by the user via theories. Here, we are interested in parts of cutlery that contact and become occluded by fruit -- this is because such parts are the relevant ones for cutting or piercing.

The python script will take as input a video and produce two folders: one of event frames, similar to the script of the 02_event_segmentation example, and one of stored_training_frames, containing pairs of images in jpg format and annotations in text files. The annotations can be used later to train segmentation models.

For an input video, you can either use your own or [the one from here](https://drive.google.com/file/d/1d9lPxgDgCQK8ijfpROWRmAq-aMjwZsLZ/view?usp=drive_link).

To run the script, you can do

```
py functional_object_segmentation.py -h
```

to get information about the command line syntax or

```
py functional_object_segmentation.py -iv <path to your video>
```

to collect the training frames and annotations for the functional object segmentation.

The format of an annotation file is the following: it is a list of lines, where each line annotates a polygon. A polygon annotation is a class id followed by space separated floating point numbers representing x and y coordinates of polygon points, normalized to the image dimensions. The first and last points in an annotation are identical.
