"""Datasets: a run's data, a batch per optimizer step.

``base.py`` holds the contracts the loops read their batches through, and
each other module the cookbook's dataset of one kind of data: tasks for the
RL loop, and conversations for the SFT loop, with what reads a file of them.
"""
