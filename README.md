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

## Basic Troubleshooting

If you come across an error when running `flask run` in the virtual environment such as:

```text
Traceback (most recent call last):
  File "/opt/homebrew/lib/python3.10/site-packages/flask/cli.py", line 245, in locate_app
    __import__(module_name)
  File "/Users/harrysharma/Desktop/test/federated-learning-web-app/app.py", line 6, in <module>
    from flask_socketio import SocketIO, emit
ModuleNotFoundError: No module named 'flask_socketio'
```

Try either:

- Restart your shell e.g. using `bash -l`, `zsh -l`, etc.
- `deactivate` followed by `source dlg-env/bin/activate` 

and try to run it again.

If this still does not fix the issue feel free to reach out.