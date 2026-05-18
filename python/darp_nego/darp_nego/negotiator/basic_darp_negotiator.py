from abc import ABC, abstractmethod
from typing import Optional, Iterable

from ..darp.basic_darp import BasicDARPClient, BasicDARPProblem
from ..outcome import BasicDARPOutcome


class BasicDARPNegotiator(ABC):

    def __init__(self, agent_id: str, problem: BasicDARPProblem):
        self.darp_problem = problem
        self.agent_id = agent_id
        self._known_clients = dict(problem.clients)
        self._client_owners = {client_id: self.agent_id for client_id in self._known_clients}
        self.initial_utility = 0
        self.current_utility = self.get_utility()
        self._last_utility_problem = None

    def update_agreement(self, outcome: BasicDARPOutcome):
        for client_id in outcome:
            owner_id = outcome[client_id]
            self._update_owner(client_id, owner_id)
        self.current_utility = self.current_utility + self.utility_change
        self.darp_problem = self._last_utility_problem
        self._last_utility_problem = None

    def _update_owner(self, client_id: int, agent_id: str):
        current_owner = self._client_owners[client_id]
        if current_owner == self.agent_id:
            self.darp_problem.remove_client(client_id)
        if agent_id == self.agent_id:
            self.darp_problem.add_client(self._known_clients[client_id])
        self._client_owners[client_id] = agent_id

    def add_client_information(self, client: BasicDARPClient, agent_id: str):
        client_id = client.client_id
        self._known_clients[client_id] = client
        self._client_owners[client_id] = agent_id

    def get_utility(self, including: Optional[Iterable[int]]=[], excluding: Optional[Iterable[int]]=[]):
        """
        Get the utility of the current state of the negotiation, 
        i.e. the cost of the clients that are included in the negotiation.
        Including and excluding clients meaning exchange of clients between agents.
        """
        if not including and not excluding:
            # Initial cost, company is not involved in any negotiation.
            # Should get in this case only when the negotiator is created.
            self.initial_utility = self.darp_problem.solve_problem()
            return self.initial_utility
        including = [self._known_clients[client_id] for client_id in including] if including else []
        excluding = [self._known_clients[client_id] for client_id in excluding] if excluding else []
        new_problem = BasicDARPProblem(self.darp_problem)
        for client in including:
            new_problem.add_client(client)
        for client in excluding:
            new_problem.remove_client(client.client_id)            
        current_cost = self.current_utility
        new_cost = new_problem.solve_problem()
        self._last_utility_problem = new_problem
        return new_cost #- current_cost

    def get_associated_cost(self, clients: Iterable[int]):
        """
        Prenegeotiation step, reveal the cost of each client 
        to decide if they should be included in the negotiation.
        """
        new_problem = self.darp_problem.copy_with_solution()
        clients_cost = 0
        for client_id in clients:
            clients_cost += new_problem.estimate_client_cost(client_id)
        del new_problem
        return clients_cost

    @abstractmethod
    def prepare_prenegotiation(self):
        pass

    @abstractmethod
    def prepare_negotiation(self):
        pass

    @abstractmethod
    def reveal_client(self) -> BasicDARPClient:
        pass

    @abstractmethod
    def propose_bid(self, **kwargs) -> BasicDARPOutcome:
        pass

    @abstractmethod
    def receive_response(self, outcome: BasicDARPOutcome, agent: str, response: bool, **kwargs):
        pass

    @abstractmethod
    def is_acceptable(self, outcome: BasicDARPOutcome, **kwargs) -> bool:
        pass

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str):
        if value:
            self._agent_id = value
        else:
            raise Exception("Agent ID should be non-empty")



