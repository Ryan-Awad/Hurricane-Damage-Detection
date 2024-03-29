# Hurricane Damage Detection
![](https://img.shields.io/badge/TensorFlow-FF6F00?&logo=TensorFlow&logoColor=fff)
![](https://img.shields.io/badge/Python-306998?&logo=Python&logoColor=FFD43B)
![](https://img.shields.io/badge/Streamlit-FF4B4B?&logo=Streamlit&logoColor=fff)
## [Submission for Waterloo's Hack the North 2020++](https://devpost.com/software/hurricane-damage-detection-rt1sz6)
### How the Model Works
Hurricanes cause lots of devastation and affect many people physically, mentally and economically. For our model, we utilized a dataset that contains satellite images of both damaged and undamaged areas in Texas after Hurricane Harvey unfortunately struck in 2017. For this reason we decided that we would create an application that would detect whether an area has been damaged by a hurricane. This would be useful for detecting if an area was damaged by a hurricane much earlier. This is great because it allows help to be sent earlier to the designated damaged area. Instead of having to wait for a damaged area to be reported, which takes longer, you could get a live satellite image of each area and see exactly which areas need help. 
### More in Depth
To build our model, we used *TensorFlow* to create a *convolutional neural network (CNN)* to detect patterns and edges in the satellite images. The optimizer used to compile model was the *Adam optimizer* with a learning rate of `1e-4`. For the loss function, we used *binary cross-entropy*. This allowed our model to achieve an accuracy of `96%`. For more info on the architecture of our model, check out the [`model-developpement`](https://github.com/Ryan-Awad/Hurricane-Damage-Detection/tree/model-developpement) branch. 
### Where We Got the Data
The data is composed of satellite images from Texas after Hurricane Harvey struck in 2017. The data is divided into 2 groups, `damage` and `no_damage`. This data allows us to train a model to detect if an area in **any** satellite image was damaged by a hurricane. The data used to train the model was taken from [here](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized).
