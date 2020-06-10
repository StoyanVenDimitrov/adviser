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

    @PublishSubscribe(sub_topics=["beliefstate"], pub_topics=["sys_act", "sys_state"])
    def simulate(self):
        """simulate dialogs with the Hancrafted policy and add them here"""
        # start a simulator and HC policy. They will post {'sys_act': sys_act, "sys_state": sys_state} and
        # {'user_acts': user_acts} or similar. Catch them and write them on the buffer. -> change the topics
        # for this function

        return {'buffer': self.buffer}

    # def store(self):
    #     """the store functionality of the experience buffer held out here"""
    #
    # def sample(self):
    #     """the sample finctionality held out"""
    #
    # def update(self):
    #     """the update finction held out here"""