## Flexible deep learning models in computer vision: Finetuing of Scene Adaptive Crowd Counting Models Using Meta learning and Network Policy Estimation
 
- I worked on one-shot scene-specific crowd counting that learns to adapt already trained model to a specific test-scene based on a single example. During finetuning different layers are freezed based on the decision of a Policy network. 

# Proposed Framework. 

- Atfirst I pre-trained a regressor model on the UCF-QNRF Dataset. It is our main crowd counting network.

- The main crowd counting network consists of a Resnet50 architecture. A Policy network is defined that determines which layers of the Resnet50 should be finetuned on the new scene environment whereas frizzing the other layers.  

- Meta Learning technique has been employed for training the model. The experiments are done on WorldExpo dataset. It has 107 separate sets of scenes each with few images of crowd samples. First 100 scenes are used to train the meta model and rest are used to test it. 
