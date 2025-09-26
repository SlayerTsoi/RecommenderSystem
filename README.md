## Create an environment

```
conda create -n Recommender_System -y
conda activate Recommender_System

```

## Install Python packages 

```
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt

```

## Run the project
```
flask --app flaskr run --debug
```

## Add the recommendation algorithm
You only need to modify the `main.py` file. Its path is as follows:
```
path: /flaskr/main.py
```