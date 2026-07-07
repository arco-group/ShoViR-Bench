from .chexagent import MODEL_CLASS as CHEXAGENT_CLASS
from .chexagent import MODEL_SPEC as CHEXAGENT_SPEC
from .gemini import MODEL_CLASS as GEMINI_CLASS
from .gemini import MODEL_SPEC as GEMINI_SPEC
from .gpt54 import MODEL_CLASS as GPT54_CLASS
from .gpt54 import MODEL_SPEC as GPT54_SPEC
from .chexone import MODEL_CLASS as CHEXONE_CLASS
from .chexone import MODEL_SPEC as CHEXONE_SPEC
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
from .radialog import MODEL_CLASS as RADIALOG_CLASS
from .radialog import MODEL_SPEC as RADIALOG_SPEC
from .llavarad import MODEL_CLASS as LLAVARAD_CLASS
from .llavarad import MODEL_SPEC as LLAVARAD_SPEC
from .spec import ModelSpec

MODEL_SPECS: dict[str, ModelSpec] = {
    GEMINI_SPEC.key: GEMINI_SPEC,
    GPT54_SPEC.key: GPT54_SPEC,
    MEDGEMMA_SPEC.key: MEDGEMMA_SPEC,
    MAIRA2_SPEC.key: MAIRA2_SPEC,
    LIBRA_SPEC.key: LIBRA_SPEC,
    CXRMATEED_SPEC.key: CXRMATEED_SPEC,
    CHEXAGENT_SPEC.key: CHEXAGENT_SPEC,
    CHEXONE_SPEC.key: CHEXONE_SPEC,
    NV_REASON_SPEC.key: NV_REASON_SPEC,
    RADIALOG_SPEC.key: RADIALOG_SPEC,
    LLAVARAD_SPEC.key: LLAVARAD_SPEC,
}

MODEL_CLASSES = {
    GEMINI_SPEC.key: GEMINI_CLASS,
    GPT54_SPEC.key: GPT54_CLASS,
    MEDGEMMA_SPEC.key: MEDGEMMA_CLASS,
    MAIRA2_SPEC.key: MAIRA2_CLASS,
    LIBRA_SPEC.key: LIBRA_CLASS,
    CXRMATEED_SPEC.key: CXRMATEED_CLASS,
    CHEXAGENT_SPEC.key: CHEXAGENT_CLASS,
    CHEXONE_SPEC.key: CHEXONE_CLASS,
    NV_REASON_SPEC.key: NV_REASON_CLASS,
    RADIALOG_SPEC.key: RADIALOG_CLASS,
    LLAVARAD_SPEC.key: LLAVARAD_CLASS,
}
