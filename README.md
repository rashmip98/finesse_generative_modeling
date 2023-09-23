# finesse_generative_modeling

## Repository Structure
|-- data  
|       &emsp;&emsp;      |-- images  
|	 &emsp;&emsp;  |-- product_data.json  
|	 &emsp;&emsp;  |-- prod_embeddings.npz  
|	 &emsp;&emsp;  |-- prod_title_map.json  
|	 &emsp;&emsp;  |-- vocab.txt  
|	 &emsp;&emsp;  |-- data_utils.py  
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
The images provided had to be scaled down to 64x64 in order to fit the compute constraints. The text data used was the title text which was first saved as a product ID to tokenized sentences json and then as a product ID to sentence embeddings in an npz map.  

## Performance  
The DC- GAN exhibited failure mode at the start with the discriminator loss going to 0, which was stabilized by adding SGD optimizer and spectral normalization. After this, the generator became too strong, with the discriminator loss increasing steadily and the generator loss decreasing to almost 0. The generator had gone in collapse mode. I removed the pixel loss and the perceptual loss from the generator's supervision, keeping only the adversarial loss training regime. The generator was still too strong. The training couldn't be completed due to my ill-health and travel before the deadline so I couldn't add further stabilization tricks to see if the training improves.

