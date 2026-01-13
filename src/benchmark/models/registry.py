from .chexagent import MODEL_SPEC as CHEXAGENT_SPEC
from .cxrmateed import MODEL_SPEC as CXRMATEED_SPEC
from .libra import MODEL_SPEC as LIBRA_SPEC
from .maira_2 import MODEL_SPEC as MAIRA2_SPEC
from .mambaxray import MODEL_SPEC as MAMBAXRAY_SPEC
from .medgemma import MODEL_SPEC as MEDGEMMA_SPEC
from .spec import ModelSpec

MODEL_SPECS: dict[str, ModelSpec] = {
    MEDGEMMA_SPEC.key: MEDGEMMA_SPEC,
    MAIRA2_SPEC.key: MAIRA2_SPEC,
    LIBRA_SPEC.key: LIBRA_SPEC,
    CXRMATEED_SPEC.key: CXRMATEED_SPEC,
    CHEXAGENT_SPEC.key: CHEXAGENT_SPEC,
    MAMBAXRAY_SPEC.key: MAMBAXRAY_SPEC,
}
