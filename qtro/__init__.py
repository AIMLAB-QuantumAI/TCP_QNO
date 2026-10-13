"""QTRO: a quantum transport-reset operator for distributional prediction of TCP/AQM dynamics.

The operator advances a probability law over discretised (window, queue, phase)
TCP states as a product of a conditioned stochastic transport, realised by a
Givens-ladder circuit traced over an environment register, and a protocol
reset that moves mass only where loss and timeout take a TCP sender.
"""

__version__ = "1.0.0"
