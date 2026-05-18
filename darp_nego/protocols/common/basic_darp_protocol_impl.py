from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ...core.darp.basic_darp import BasicDARPNegotiationProblem, BasicDARPClient
from ...core.negotiator import BasicDARPNegotiator
from ...core.outcome import BasicDARPOutcome


class BasicDARPMechanism(ABC):

    def __init__(self, agents=Optional[Iterable[BasicDARPNegotiator]], **kwargs):
        self.participants = {agent.agent_id: agent for agent in agents} if agents else {} # all agents (companies), maybe in future we can choose which companies will participate
        self.agreement_history: Iterable[(int, BasicDARPOutcome)] = []

    def add_participant(self, agent: BasicDARPNegotiator, **kwargs):
        self.participants[agent.agent_id] = agent

    def remove_participant(self, agent_id: str):
        if agent_id in self.participants:
            del self.participants[agent_id]


    @abstractmethod
    def prenegotiation(self):
        pass

    @abstractmethod
    def round(self):
        pass

    @abstractmethod
    def negotiation(self):
        pass
