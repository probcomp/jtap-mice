from typing import NamedTuple
from jtap.inference import JTAPData
from .beliefs import JTAP_Beliefs
from .decisions import JTAP_Decisions
from .metrics import JTAP_Metrics

class JTAP_Results(NamedTuple):
    jtap_data: JTAPData
    jtap_beliefs: JTAP_Beliefs
    jtap_decisions: JTAP_Decisions
    jtap_metrics: JTAP_Metrics