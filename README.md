# federated-learning-results

## How to run locally

1. Make sure to have [Python](https://www.python.org/downloads/) on your system.
2. Clone this repo using: 
```
git clone https://github.com/harrysharma1/federated-learning-results.git
```
3. Create virtual environment and activate environment. Either:
```
$ virtualenv dlg-env
$ source dlg-env/bin/activate
(dlg-env)$
```
[`virtualenv` can be found here](https://pypi.org/project/virtualenv/#description). Or, you can use:
```
$ python3 -m venv dlg-env
$ source dlg-env/bin/activate
(dlg-env)$
```
Provided with `python3`.
4. Download requirements with:
```
pip install -r requirements.txt
```
5. Run flask development server:
```
flask run
```