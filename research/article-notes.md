# research article findings

## Recognizing Manipulation Actions from State-Transformations

### What they do: 
* Object state transitions for as means of recognising manipulation actions

### How they tackle it: 
* Object states more obvious/apparent than actions, therefore use state transition matrix that maps actions into pre- and post-state, using keyframes. Learn appearance models of objects/states, thereby letting manipulation actions be recognised from the matrix.

### limitations/scope: 
* Only RGB channels used. Only 20 million parameters and 5m trainable parameters - much lower than baseline techniques (still comparable to other models)

## Long-Term Feature Banks for Detailed Video Understanding

What they do: long term feature bank - includes details of past/future objects, actions and scenes in a movie. 

How they tackle it: use pre-computed visual features - allows long sections of video unlike most video models that can only make predictions from short spans of videos since they use 3D convolutions (require dense sampling). Long-term bank as auxiliary component allows storage of flexible storage info.

limitations/scope: used for three datasets (AVA, EPIC-Kitchens, Charades)

## Object Level Visual Reasoning in Videos

What they do: model that recognises/reasons about spatio-temporal interactions. Reasoning done at object level 

How they tackle it: reasoning done at object level via object detection networks. Object detection to extract low level information from scene, map interactions and objects themselves using Object Relation Network. 

limitations/scope: three standard datasets (Twenty-BN Something-Something, VLOG and EPIC Kitchens)
