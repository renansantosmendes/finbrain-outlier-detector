"""Train and serve an Isolation Forest that flags atypical market days.

The package has four layers: ``data`` loads prices, ``features`` turns them
into model inputs, ``models`` fits and stores the detector, and ``serving``
exposes the trained detectors through a FastAPI application.
"""
