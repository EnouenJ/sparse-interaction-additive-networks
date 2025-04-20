from sian.interpret.aggregate_detections import (
    aggregateDetections_only1D,
    aggregateDetections_only2D,
    aggregateDetections_anyD,

    aggregateContrastiveDetections_only1D,
    aggregateContrastiveDetections_only2D,
    aggregateContrastiveDetections_anyD,

    aggregateConditionalDetections_only1D,
    aggregateConditionalDetections_only2D,
    aggregateConditionalDetections_only1D_V2,
    aggregateConditionalDetections_only2D_V2,

    aggregateGroupedContrastiveDetections_only1D,
    aggregateGroupedContrastiveDetections_only2D,
    aggregateGroupedContrastiveDetections_anyD,

    aggregateGroupedDetections_anyD,
)

from sian.interpret.basic_wrappers import (
    ModelWrapperTorch,
    MixedModelWrapperTorch,
    MixedModelEnsembleWrapperTorch,
    BloodMixedEnsembleWrapperTorchLogit,
    SKlearnEnsembleWrapperLogit,
    BasicXformer,
    CustomGroupedXformer, #CustomizedGroupedXformer,

    MaskedXformer,
    MaskedXformer_v2,
    Masked_MixedModelWrapperTorch,
)

# from sian.interpret.explainer import Archipelago #NOTE: disabled on 04/12/2025 @ 11:00pm

from sian.interpret.notebook_utils import prettyPrintInteractionSingles, prettyPrintInteractionPairs
from sian.interpret.notebook_utils import prettyPrintInteractions



from sian.interpret.explainer2 import JamArchipelago
from sian.interpret.explainer2 import JamMaskedArchipelago
from sian.interpret.explainer2 import unmasked_FID_Hyperparameters
from sian.interpret.explainer2 import masked_FID_Hyperparameters




from sian.interpret.plotting.plotting import (
    plot_1D_log_log_interaction_histogram,
    plot_2D_log_log_interaction_histogram,

    fancy_plot_archipelago_covariances,
)



from .plotting.shape_plotting import plot_all_sinewaves_from_synthetic
from .plotting.shape_plotting import plot_all_raw_PDPs

from .plotting.gam_shape_plotting import plot_all_GAM_functions

