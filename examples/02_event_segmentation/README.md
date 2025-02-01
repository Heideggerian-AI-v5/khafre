This example showcases "image schematic event segmentation".

Image schemas are a notion coming from cognitive linguistics, and represent basic sensori-motor patterns of experience postulated to form the basis of human concepts. Examples of image schemas are containment, support, linkage, contact, verticality.

Note that, perhaps somewhat confusingly, image schemas do not necessarily refer to images; rather, they are patterns that can involve any kind of sensory data, including auditory and haptic. However, with khafre we can only look at visual clues.

By image schematic segmentation, we mean detecting times at which some image schematic relation changes, e.g. some objects come into contact, or stop being in contact. The python script will take as input a video and produce a folder of "event frames", which are pairs of a jpg image corresponding to the frame and a triples file in Turtle format describing what relations changed and how. The script will also produce a summary.html file which provides a visualization of the timeline of image schematic events.

This example uses a YOLO model pretrained on the COCO dataset. We instruct Khafre via the theories to look for changing movement, contact, and linkage relations between agents (persons and various animals in the COCO classes). You can find a video you could try with the script [at this address](https://drive.google.com/file/d/1uPvNIDJp3k-sNgjjeZiEOGumdtbvztHi/view?usp=drive_link) or try your own.

To start the script, you can run

```
py image_schematic_summary.py -h
```

to get info on the command line syntax or

```
py image_schematic_summary.py -iv <path to your video>
```

to run the image schematic segmentation example.
