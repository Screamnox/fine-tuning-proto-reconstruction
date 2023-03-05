# mBART fine-tuned for proto-Latin reconstruction
Purpose: to fine-tune the mBART model for the reconstruction of Proto-Latin from descending Romance languages.

*The database used is from [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

## Running Locally

###  Cloning the repository the local machine.

```bash
git clone https://github.com/Screamnox/fine-tuning-mBART
```

### Installing the dependencies.

Start by creating a virtual environment with `pip` in your project directory, and activate the virtual environment.

- Then :

```bash
pip install -r requirements.txt
```
*Currently requirements.txt is not available.*

- **OR :** if you don't want to install all the libraries :

Install [PyTorch](https://pytorch.org/get-started/locally/) and the Hugging Face Transformers library via `pip install transformers`.<br>
(And the datasets library : `pip install datasets`) 

### Running the training

```bash
py test.py
```
*Currently not working.*
