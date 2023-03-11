# mBART fine-tuned for proto-Latin reconstruction
Purpose: to fine-tune the multilingual model for the reconstruction of Proto-Latin from descending Romance languages.

*The database used is from [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

## How it works?

- **Initialization**
  - Initialize the model and its tokenizer on the desired device (cpu/gpu).
- **Data processing**
  - Loads the database in a format understandable to the Transformers library.
  - Split into three sets: train, validation and test. (But one and the same object at the end of the separation.)
  - Tokenize all the words contained in the preprocessed database, and put it in a format understandable to PyTorch.
  - Load individually the sets (training, test, validation) by sampling them in batches. (Checks that all the processing on the data is properly done.)
- **Training**
  - Initialize the training parameters : an optimizer, a number of epochs, a scheduler. 
  - Start the training and evaluation loop

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
