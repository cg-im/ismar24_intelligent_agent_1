# ismar24_intelligent_agent_1

### Toward User-Aware Interactive Virtual Agents: Generative Multi-Modal<br> Agent Behaviors in VR

##Intelligent agent conversation system

### System Configuration
Unreal Engine version: 5.3

Required Varjo Aero VR headset to run the system. Following link provide you with steps to setup the varjo aero headset.
https://varjo.com/use-center/get-started/varjo-headsets/setting-up-your-headset/setting-up-aero/

Steam VR Traking device for Headset and Motion contrller tracking.
[https://varjo.com/use-center/get-started/varjo-headsets/setting-up-your-headset/setting-up-aero/](https://varjo.com/use-center/get-started/varjo-headsets/setting-up-tracking/steamvr-tracking/)


### CVAE model
Our CVAE model code is included in the CVAE directory which also include pretrained weights in ONNX format. 
You can also trin the model using the code provided. For this you will need to request the dataset via the project page.

### VR Conversation Agent system
AnimSynth directory contains our VR agent conversation system. 


![Teaser](https://github.com/user-attachments/assets/4a66fddc-3107-4a74-947b-680b0c8d460d)

### Abstract 
Virtual agents serve as a vital interface within XR platforms. 
However, generating virtual agent behaviors typically rely on pre-coded
actions or physics-based reactions. In this paper we present a
learning-based multimodal agent behavior generation framework
that adapts to users’ in-situ behaviors, similar to how humans 
interact with each other in the real world. By leveraging an in-house
collected, dyadic conversational behavior dataset, we trained a 
conditional variational autoencoder (CVAE) model to achieve user 
conditioned generation of virtual agents’ behaviors. Together with
large language models (LLM), our approach can generate both the
verbal and non-verbal reactive behaviors of virtual agents. Our
comparative user study confirmed our method’s superiority over 
conventional animation graph-based baseline techniques, particularly
regarding user-centric criteria. Thorough analyses of our results
underscored the authentic nature of our virtual agents’ interactions
and the heightened user engagement during VR interaction.

Fill [this form](https://forms.gle/SoRcmLBgWxQqiyoo8) to request the dataset used in this work. 

**Full References:** 

Toward Socially-Aware Interactive Virtual Agents: Generative Multi-Modal Agent Behaviors in VR
Bhasura S. Gunawardhana, Yunxiang Zhang, Qi Sun, and Zhigang Deng,
Proceedings of 23rd IEEE International Symposium on Mixed and Augmented Reality (ISMAR) 2024, Seattle, USA, Oct 21- 25, 2024.

BibTeX
```
@INPROCEEDINGS{virtualAgentBehaviours,
  author={Bhasura S. Gunawardhana, Yunxiang Zhang, Qi Sun, and Zhigang Deng},
  booktitle={2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)}, 
  title={Toward User-Aware Interactive Virtual Agents: Generative Multi-Modal Agent Behaviors in VR}, 
  year={2024},
  volume={},
  number={},
  pages={},
  keywords={Virtual agents, human-VR interaction, user-conditioned motion generation, user-aware interaction.},
  doi={}}
```

