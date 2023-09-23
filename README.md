# lip-reading
LRS3-based lip reading algorithm using Inception3D features &amp; self-attention. Handles 4 segments of 16 frames for word prediction. 

This GitHub repository is focused on a lip reading algorithm developed using the LRS3 dataset, which comprises spoken sentences from TED and TEDX videos. Each video is centered on the speaker's face, and there are corresponding text transcripts for the videos. The dataset is divided into three sets: Pre-train, Trainval, and Test.

The lip reading algorithm implemented in this repository follows a multi-step approach:

Feature Extraction: It uses the Inception3D model, pre-trained and available in the GitHub repository, to extract features from 16-frame segments of the videos.
Self-Attention: The model applies self-attention to the extracted features. It computes 400-dimensional feature vectors for each 16-frame segment, repeated for four different segments.
Vocabulary Prediction: Depending on the size of the vocabulary, the model further processes these 400-dimensional vectors with self-attention. These processed vectors are used to predict words from the vocabulary, with the word having the highest activation value considered as the prediction.
The specific focus of this lip reading model is to identify single words in short video segments, and it acknowledges that not many video segments in the LRS3 dataset fit this specific formulation due to varying video durations and word lengths. However, the repository does provide some examples of video segments that do meet this formulation, which are used for both training and evaluation of the model.

In summary, this GitHub repository houses code for a lip reading model designed to work with the LRS3 dataset, using a combination of pre-trained Inception3D features and self-attention mechanisms to predict words from mouth-centered video segments, particularly those consisting of 16-frame segments with a focus on single words.

Team Members:
 - Srinath Dittakavi
 - Mohamed Mehdi Bourahla
 - Gagan Jagadish
 - Cade Mack
 - Varun Upadhyay
