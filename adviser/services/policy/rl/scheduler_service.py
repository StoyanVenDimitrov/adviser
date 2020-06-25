from services.service import Service, PublishSubscribe
from utils.domain.jsonlookupdomain import JSONLookupDomain



class Scheduler(Service):
    """Used for scheduling the policies thorough the belief state tracking"""
    def __init__(
            self,
            domain: JSONLookupDomain,
            switch_policies
    ):
        Service.__init__(self, domain=domain)
        self.switch_after = switch_policies
        self.num_of_dialogs = 0


    def dialog_start(self):
        self.num_of_dialogs += 1

    @PublishSubscribe(sub_topics=["beliefstate"], pub_topics=["beliefstate_rl", "beliefstate_hcp"])
    def bst_scheduler(self, beliefstate):
        if self.num_of_dialogs <= self.switch_after:
            return {"beliefstate_hcp": beliefstate}
        if self.num_of_dialogs > self.switch_after:
            return {"beliefstate_rl": beliefstate}