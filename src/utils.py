import warnings


def suppress_warnings():
    """Suppress unessential warnings."""
    warnings.filterwarnings("ignore")
    print("Unessential warnings suppressed.")
