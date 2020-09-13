import torch
from services.policy.rl.experience_buffer import NaivePrioritizedBuffer, UniformBuffer
from services.policy.rl.policy_rl import RLPolicy
from services.service import Service, PublishSubscribe
from services.stats.evaluation import ObjectiveReachedEvaluator
from utils import BeliefState, UserActionType, SysAct, SysActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger


class Buffer(Service, RLPolicy):
    """Experience buffer as a service"""
    # TODO: remove hard coded parameter values
    def __init__(
            self,
            domain: JSONLookupDomain,
            buffer_classname,
            batch_size=64,
            buffer_size=6000,
            training_frequency=20,
            logger: DiasysLogger = DiasysLogger(),
            device=torch.device('cpu')
    ):
        Service.__init__(self, domain=domain)

        RLPolicy.__init__(
            self,
            domain, buffer_cls=buffer_classname,
            buffer_size=buffer_size, batch_size=batch_size,
            discount_gamma=None, include_confreq=None,
            logger=logger, max_turns=25, device=device)
        self.is_training = True
        self.total_train_dialogs = 0
        self.training_frequency = training_frequency
        self.atomic_actions = ["inform_byname",  # TODO rename to 'bykey'
                               "inform_alternatives",
                               "reqmore",
                               'closingmsg']
        self.turns = 0
        self.unlock_buffer = True

    def dialog_start(self, dialog_start=False):
        self.sys_state = {
            "lastInformedPrimKeyVal": None,
            "lastActionInformNone": False,
            "offerHappened": False,
            'informedPrimKeyValsSinceNone': []}
        self.turns = 0

    @PublishSubscribe(sub_topics=["beliefstate", "sys_act"])
    def store(self, beliefstate, sys_act):#, next_id):
        self.turns += 1
        """the store functionality of the experience buffer held out here"""
        if sys_act == self.hello_action(): # or sys_act == self.bye_action():
            return self.do_sample()
        if self.turns > self.max_turns:
            return self.do_sample()
        try:
            action_id = self.make_action_id_from_action(sys_act)
        except ValueError:
            return

        state_vector = self.beliefstate_dict_to_vector(beliefstate)
        self.turn_end(beliefstate, state_vector, action_id)
        return self.do_sample()

    #@PublishSubscribe(pub_topics=["training_batch"])
    def dialog_end(self):
        self.end_dialog(self.sim_goal)
        self.total_train_dialogs += 1
        if self.writer is not None:
            self.writer.add_scalar('buffer/items', len(self.buffer),
                            self.total_train_dialogs)

    @PublishSubscribe(pub_topics=["training_batch"])
    def do_sample(self):
        if len(self.buffer) >= self.batch_size * 10 and \
                self.total_train_dialogs % self.training_frequency == 0:
            if self.unlock_buffer:
                batch = self.buffer.sample()
                self.unlock_buffer = False
                return {'training_batch': batch}
            else:
                return {'training_batch': None}
        else:
            self.unlock_buffer = True
            return {'training_batch': None}

    @PublishSubscribe(sub_topics=["sim_goal"])
    def end(self, sim_goal):
        self.sim_goal = sim_goal

    @PublishSubscribe(sub_topics=["buffer_update"])
    def update(self, buffer_update):
        """Carries out a buffer update"""
        for elem in buffer_update:
            # self.buffer.update(elem(0), elem(1)) leads to python error,
            # but it wasn't reported, only the results were very bad
            self.buffer.update(elem[0], elem[1])

    def make_action_id_from_action(self, action):
        slot_values = action.slot_values
        name = action.__str__().split('(')[0]
        if not slot_values or name in self.atomic_actions:
            action_name = name
        else:
            action_name = name + '#' + list(slot_values.keys())[0]
        return self.actions.index(action_name)

    @staticmethod
    def hello_action():
        sys_act = SysAct()
        sys_act.type = SysActionType.Welcome
        return sys_act

    @staticmethod
    def bye_action():
        sys_act = SysAct()
        sys_act.type = SysActionType.Bye
        return sys_act
