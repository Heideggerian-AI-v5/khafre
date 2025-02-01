# khafre -- Knowledge-based Heideggerian AFfordance REcognition

Perception for affordance recognition and learning from observed interactions.

KHAFRE combines computer vision techniques -- classical and machine learning based -- with reasoning to produce a taskable perception system for analyzing objects and events into functional parts. For objects, functional parts are those with particular relevance for the performing of particular actions; examples would be the blade of a knife, or its handle. For events, functional parts refer to subevents at which qualitative relations between objects change; qualitative relations are those such as contacts, occlusions, states of relative movement etc., from which more abstract relations such as support, linkage, or containment could be inferred. 

This analysis is powered by "theories", i.e. it is possible to describe to the system what kinds of behaviors to look for, and it then selects object and event regions based on that description. See the examples folders 02_event_segmentation and 03_functional_object_segmentation for more information.

## Dependencies and installation

Currently, limited to python versions 3.9.x to 3.12.x. Newer versions of python will be supported as soon as pytorch is available for them.

Tested on Windows and Linux systems. MacOS will not work at the moment because the python multiprocessing package is not fully implemented on that OS.

It is recommended to install inside a virtual environment. The easiest way is to git clone the repository, start the virtual environment, and run the install.py script inside of it.

Alternatively, one can git clone the repository and run (installing as editable is optional but recommended)

```
pip3 install transformers==4.40
pip3 install -e .
```

Assuming the pip3 will know to install pytorch with CUDA support, the above pair is sufficient. Installing pytorch without CUDA support should still work, but will be significantly slower. 

TEMPORARY: we prefer not to force packages to particular versions, however versions of transformers from 4.41 onwards introduced an API bug in their depth estimator module. Until a release fixes it, we therefore lock to this version. 

## Using khafre

Khafre is intended to be imported by a larger program; the examples folders show the intended usage pattern. 

The basic concepts in khafre are the "brick" and the "wire". Bricks are processes that can run in parallel (avoiding python's Global Interrupt Lock) and each perform a specialized function, such as video input, or displaying debug images, or tracking objects etc. Wires are communication channels between bricks, and have two components: a notification queue on which some arbitrary format data may be exchanged, and a shared memory buffer on which a fixed size array can be shared. The notification queue should be used for relatively small data, with the shared buffer being intended for images.

Hotwiring bricks is not possible: once a brick is started, its incoming/outgoing wiring must not be changed.

Once a brick is started however it may receive commands to further complete initialization, or (re)set parameters, or as part of the normal operation of the brick, e.g., to register what sort of queries the brick should answer next. The latter kind of command is usually sent by a specialized brick, a "reasoner". The reasoner is configurable by way of "theories" in defeasible logic. More information on that logic is available in our repository about a [defeasible logic implementation](https://github.com/mpomarlan/silkie).
