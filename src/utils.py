import warnings


def suppress_warnings():
    """Suppress all Python runtime warnings."""
    warnings.filterwarnings("ignore")
    print("Unessential warnings suppressed.")
