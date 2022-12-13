# metaphor

* identifying metaphors in sentences
* we were initially interested in viral language
* BERT as main starting point
  * Building on MelBERT model
* Roberta added reddit for more training, so many leaning this way
* add a masked model - calculates the probability of a token occuring at any given point

Experiments
1. 2 langauge heads, every single token in training set has a 0 or 1 for metphor label. 
* translated labeled toekn into new token that incorporated metaphor label
* shifted index based on label 
* INNOVATION --> moved from sequence classifaction task (MelBERT) to a token level classification 
* classify each token as a metaphor or not on left encoder (see MelBERT)

# Tasks
* Look at MERMAID paper
* look at labeling of dataset
