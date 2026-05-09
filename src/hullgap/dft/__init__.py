"""
Targeted DFT validation on top of MLIP screening.

DFT is a follow-up layer: it does not replace high-throughput MLIP relaxation
or hull ranking. Only a small set of top MLIP-ranked structures receive VASP
inputs; results are meant for prioritization and later higher-accuracy study,
not as final claims of thermodynamic stability.
"""
