# finesse_generative_modeling

## Repository Structure
|-- data  
|       &msp;&msp;      |-- images  
|	 &msp;&msp;  |-- product_data.json  
|	 &msp;&msp;  |-- prod_embeddings.npz  
|	 &msp;&msp;  |-- prod_title_map.json  
|	 &msp;&msp;  |-- vocab.txt  
|	 &msp;&msp;  |-- data_utils.py  
|-- model.py  
|-- dataset.py  
|-- train.py  
|-- inference.py  
|-- config  
|-- requirements.txt  
|-- chkpts -> saved_models  
|-- results -> saved images  

## Model Details  
The architecture uses the Generative Adversarial Text to Image Synthesis paper as a rough guide but with a few architectural and technical changes. The changes have been explained in the document. The model I've used is a DC-GAN which uses text embeddings as a control input to generate images.
![image](https://github.com/rashmip98/finesse_generative_modeling/assets/31537022/4d23ba67-c266-41a8-9264-d3ca89974b8d)

## Training Data  
The images provided had to be scaled down to 64x64 in order to fit the compute constraints. The text data used was the 


