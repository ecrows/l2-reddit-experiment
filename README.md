# l2-reddit-experiment

Code for the submitted paper *"Towards the Ethical Detection of Online Influence Campaigns"* by Crothers et al. (2019).

Based off of the [official BERT experiment code](https://github.com/google-research/bert).

**Abstract**

The detection of clandestine efforts to influence users in online communities is a challenging problem with significant active development.  We demonstrate that features derived from the text of user comments are useful for identifying suspect activity, but lead to increased erroneous identifications (false positive classifications) when keywords over-represented in past influence campaigns are present.  Drawing on research in native language identification (NLI), we use "named entity masking" (NEM) to create sentence features robust to this shortcoming without a reduction in classification accuracy.  We demonstrate that while NEM consistently reduces false positives when key named entities are mentioned, both masked and unmasked models exhibit increased false positive detection on English sentences by Russian native speakers, raising ethical considerations that should be addressed in future research.

**Instructions**

Install dependencies using:
`pip3 install -r requirements.txt`

Code can then be executed using the provided helper scripts, or the run_experiment.py file can be called directly.

**Requirements**

This code is written to be run on a TPU in Google Cloud, but can also be run by CPU/GPU machines (however, this will likely require minor modifications to use the BERT-Base model to fit memory limits).
