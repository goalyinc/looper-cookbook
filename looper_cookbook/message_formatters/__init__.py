"""Message formatters: each model family's conversations as its tokens, and
its sampled tokens back as messages.

``base.py`` holds the ``MessageFormatter`` contract, and each other module one
family's format, written by hand from the model's chat template. ``models.py``
names the formatter of each model Looper serves.
"""
