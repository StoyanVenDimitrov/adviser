from services.policy.rl.experience_buffer import NaivePrioritizedBuffer, UniformBuffer
from services.service import Service, PublishSubscribe
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger


class Buffer(Service):
    """Experience buffer as a service"""

    def __init__(self, domain: JSONLookupDomain, buffer_classname, logger: DiasysLogger = DiasysLogger()):
        Service.__init__(self, domain=domain)
        if buffer_classname == "prioritized":
            self.buffer = NaivePrioritizedBuffer
        elif buffer_classname == "uniform":
            self.buffer = UniformBuffer
    @PublishSubscribe()
    def simulate(self):
        """simulate dialogs with the Hancrafted policy and add them here"""

        return {'buffer': self.buffer}