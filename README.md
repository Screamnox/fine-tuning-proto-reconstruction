# mBART fine-tuned for proto-Latin reconstruction
Purpose: to fine-tune the multilingual model for the reconstruction of Proto-Latin from descending Romance languages.

*The database used is from [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

## Running Locally

1. Cloning the repository on the local machine.
2. Installing the dependencies.
 - Start by creating a virtual environment with `pip` in your project directory, and activate the virtual environment.
 - Then, install the libraries with the file `requirement.txt` *(currently not available)* :
 ```bash 
 pip install -r requirements.txt
 ```
 
 - **OR** install them manually :<br> 
 Install [PyTorch](https://pytorch.org/get-started/locally/) and the Hugging Face Transformers library via `pip install transformers`.<br>
 (And the datasets library : `pip install datasets`) 

3. Running the training *(currently not working)* :
```bash
py test.py
```
