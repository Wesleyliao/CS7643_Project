import tqdm

def is_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell in ['ZMQInteractiveShell', 'Shell']:
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_pbar(*args, **kwargs):
    if is_jupyter_notebook():
        return tqdm.tqdm_notebook(*args, **kwargs)
    else:
        return tqdm.tqdm(*args, **kwargs)