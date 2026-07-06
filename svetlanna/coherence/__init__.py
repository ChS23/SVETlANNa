"""SVETlANNa partial spatial coherence module.

Usage
-----
    from svetlanna.coherence import (
        PartiallyCoherentWavefront,
        gaussian_schell_model,
        modes_from_coherence_matrix,
        schell_model,
        SpatialCoherence,
        coherence_forward,
        sample_realizations,
        incoherent_source,
        detect_realizations,
    )
"""

from .wavefront_pc import PartiallyCoherentWavefront, _make_sim_params_modes
from .sources import (
    gaussian_schell_model,
    modes_from_coherence_matrix,
    schell_model,
    FilipoviCoherenceSource,
)
from .gamma import SpatialCoherence, coherence_forward
from .sampling import sample_realizations, incoherent_source, detect_realizations
from .assess import assess, SourceSpec, CoherenceAssessment

__all__ = [
    "PartiallyCoherentWavefront",
    "gaussian_schell_model",
    "modes_from_coherence_matrix",
    "schell_model",
    "FilipoviCoherenceSource",
    "SpatialCoherence",
    "coherence_forward",
    "sample_realizations",
    "incoherent_source",
    "detect_realizations",
    "assess",
    "SourceSpec",
    "CoherenceAssessment",
]
