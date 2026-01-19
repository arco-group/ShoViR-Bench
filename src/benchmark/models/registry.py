from .chexagent import MODEL_CLASS as CHEXAGENT_CLASS
from .chexagent import MODEL_SPEC as CHEXAGENT_SPEC
from .cxrmateed import MODEL_CLASS as CXRMATEED_CLASS
from .cxrmateed import MODEL_SPEC as CXRMATEED_SPEC
from .libra import MODEL_CLASS as LIBRA_CLASS
from .libra import MODEL_SPEC as LIBRA_SPEC
from .maira_2 import MODEL_CLASS as MAIRA2_CLASS
from .maira_2 import MODEL_SPEC as MAIRA2_SPEC
from .medgemma import MODEL_CLASS as MEDGEMMA_CLASS
from .medgemma import MODEL_SPEC as MEDGEMMA_SPEC
from .nv_reason_cxr import MODEL_CLASS as NV_REASON_CLASS
from .nv_reason_cxr import MODEL_SPEC as NV_REASON_SPEC
from .spec import ModelSpec

MODEL_SPECS: dict[str, ModelSpec] = {
    MEDGEMMA_SPEC.key: MEDGEMMA_SPEC,
    MAIRA2_SPEC.key: MAIRA2_SPEC,
    LIBRA_SPEC.key: LIBRA_SPEC,
    CXRMATEED_SPEC.key: CXRMATEED_SPEC,
    CHEXAGENT_SPEC.key: CHEXAGENT_SPEC,
    NV_REASON_SPEC.key: NV_REASON_SPEC,
}

MODEL_CLASSES = {
    MEDGEMMA_SPEC.key: MEDGEMMA_CLASS,
    MAIRA2_SPEC.key: MAIRA2_CLASS,
    LIBRA_SPEC.key: LIBRA_CLASS,
    CXRMATEED_SPEC.key: CXRMATEED_CLASS,
    CHEXAGENT_SPEC.key: CHEXAGENT_CLASS,
    NV_REASON_SPEC.key: NV_REASON_CLASS,
}
