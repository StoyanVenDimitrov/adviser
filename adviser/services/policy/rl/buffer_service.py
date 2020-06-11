import torch

from services.policy.rl.experience_buffer import NaivePrioritizedBuffer, UniformBuffer
from services.service import Service, PublishSubscribe
from services.stats.evaluation import ObjectiveReachedEvaluator
from utils import BeliefState, UserActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger


class Buffer(Service):
    """Experience buffer as a service"""
    # TODO: remove hard coded parameter values
    def __init__(
            self,
            domain: JSONLookupDomain,
            buffer_classname,
            batch_size=64,
            buffer_size=6000,
            state_dim=298,  # hard-coded for now...
            logger: DiasysLogger = DiasysLogger()
    ):
        Service.__init__(self, domain=domain)
        if buffer_classname == "prioritized":
            self.buffer = NaivePrioritizedBuffer(buffer_size,batch_size,state_dim)
        else:
            self.buffer = UniformBuffer(buffer_size,batch_size,state_dim)
        # needed to build state vector:
        self.sys_state = None
        self.evaluator = ObjectiveReachedEvaluator(domain, logger=logger)

    # @PublishSubscribe(sub_topics=["beliefstate"], pub_topics=["buffer"])
    # def simulate(self):
    #     """simulate dialogs with the Hancrafted policy and add them here"""
    #     # start a simulator and HC policy. They will post {'sys_act': sys_act, "sys_state": sys_state} and
    #     # {'user_acts': user_acts} or similar. Catch them and write them on the buffer. -> change the topics
    #     # for this function
    #
    #     return {'buffer': self.buffer}

    @PublishSubscribe(sub_topics=["beliefstate", "sys_act", "sys_state", "sim_goal"], pub_topics=["buffer"])
    def store(self, beliefstate, sys_act, sys_state, sim_goal):

        print("Store:")
        """the store functionality of the experience buffer held out here"""
        # sys_state comes after update. Actually, the sys_state from a step earlier is needed:
        try:
            state_vector = self.beliefstate_dict_to_vector(beliefstate)
        # skipping start and end of dialog, as dqnpolicy do:
        except (TypeError, KeyError):
            self.sys_state = sys_state
            return

        # append action name and when training, convert it to action_id
        # instead of complex ways to convert it before storing in the buffer.
        action = sys_act
        reward = self.evaluator.get_turn_reward()
        # term = None
        self.buffer.store(state_vector, action, reward, False)
        self.sys_state = sys_state
        return {'buffer': self.buffer}

    @PublishSubscribe( pub_topics=["buffer"])
    def dialog_end(self):
        final_reward, success = self.evaluator.get_final_reward(sim_goal=self.sim_goal)
        self.buffer.store(None, None, final_reward, terminal=True)

        return {'buffer': self.buffer}

    @PublishSubscribe(sub_topics=["sim_goal"])
    def end(self, sim_goal):
        """
            Once the simulation ends, need to store the simulation goal for evaluation

            Args:
                sim_goal (Goal): the simulation goal, needed for evaluation
        """
        self.sim_goal = sim_goal

    def beliefstate_dict_to_vector(self, beliefstate: BeliefState) -> torch.FloatTensor:
        """ Converts the beliefstate dict to a torch tensor

        Args:
            beliefstate: dict of belief (with at least beliefs and system keys)

        Returns:
            belief tensor with dimension 1 x state_dim
        """

        belief_vec = []

        # add user acts
        belief_vec += [1 if act in beliefstate['user_acts'] else 0 for act in UserActionType]
        # handle none actions
        belief_vec.append(1 if sum(belief_vec) == 0 else 1)

        # add informs (including special flag if slot not mentioned)
        for slot in sorted(self.domain.get_informable_slots()):
            values = self.domain.get_possible_values(slot) + ["dontcare"]
            if slot not in beliefstate['informs']:
                # add **NONE** value first, then 0.0 for all others
                belief_vec.append(1.0)
                # also add value for don't care
                belief_vec += [0 for i in range(len(values))]
            else:
                # add **NONE** value first
                belief_vec.append(0.0)
                bs_slot = beliefstate['informs'][slot]
                belief_vec += [bs_slot[value] if value in bs_slot else 0.0 for value in values]

        # add requests
        for slot in sorted(self.domain.get_requestable_slots()):
            if slot in beliefstate['requests']:
                belief_vec.append(1.0)
            else:
                belief_vec.append(0.0)

        # append system features
        belief_vec.append(float(self.sys_state['lastActionInformNone']))
        belief_vec.append(float(self.sys_state['offerHappened']))
        candidate_count = beliefstate['num_matches']
        # buckets for match count: 0, 1, 2-4, >4
        belief_vec.append(float(candidate_count == 0))
        belief_vec.append(float(candidate_count == 1))
        belief_vec.append(float(2 <= candidate_count <= 4))
        belief_vec.append(float(candidate_count > 4))
        belief_vec.append(float(beliefstate["discriminable"]))

        # convert to torch tensor
        return torch.tensor([belief_vec], dtype=torch.float)
