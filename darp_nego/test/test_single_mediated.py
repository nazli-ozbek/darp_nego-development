import unittest
from darp_nego.protocols.learning.single_mediated_text import SingleMediatedTextMechanism
from darp_nego.core.darp import BasicDARPClient, BasicDARPVehicle, BasicDARPProblem
from darp_nego.core.negotiator import BasicDARPNegotiator
from darp_nego.core.outcome import BasicDARPOutcome, DARPNegotiationDomain


class TestSingleMediatedTextMechanism(unittest.TestCase):

    def setUp(self) -> None:

        self.smtp = SingleMediatedTextMechanism()
        self.client1 = BasicDARPClient(0, 0, 100, 20, 30, 50, 60, 3)
        self.client2 = BasicDARPClient(1, 0, 100, 20, 30, 50, 60, 3)
        self.client3 = BasicDARPClient(2, 0, 100, 20, 30, 50, 60, 3)
        self.client4 = BasicDARPClient(3, 0, 100, 20, 30, 50, 60, 3)
        self.client5 = BasicDARPClient(4, 0, 100, 20, 30, 50, 60, 3)
        self.client6 = BasicDARPClient(5, 0, 100, 20, 30, 50, 60, 3)
        self.client7 = BasicDARPClient(6, 0, 100, 20, 30, 50, 60, 3)
        self.domain1 = DARPNegotiationDomain()
        self.domain1.add_client(self.client1, "agent1")
        self.domain1.add_client(self.client2, "agent1")
        self.domain1.add_client(self.client3, "agent2")
        self.domain1.add_client(self.client4, "agent3")
        self.domain1.add_client(self.client5, "agent4")
        self.domain1.add_client(self.client6, "agent4")
        self.domain1.add_client(self.client7, "agent4")
        self.smtp.domain = self.domain1

    def test_graph1(self):
        outcome = {0: "agent1", 1: "agent1", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent4", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent3", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.edges), 0)
        self.assertEqual(len(g.nodes), 0)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertFalse(ac)

    def test_graph2(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent4", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent3", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 1)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_edge("agent2", "agent1", 0))

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {1: "agent2"})


    def test_graph3(self):

        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent2", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_node("agent4"))
        self.assertTrue(g.has_edge("agent4", "agent1"))
        self.assertEqual(g.get_edge_data("agent4", "agent1")[0]["client"], 1)
        self.assertEqual(g.get_edge_data("agent2", "agent4")[0]["client"], 5)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {1: "agent4", 5: "agent2"})

    def test_graph4(self):
        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent2", 6: "agent4"}
        positive_responses = set(["agent1", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 1)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {})

    def test_graph5(self):
        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent2", 6: "agent4"}
        positive_responses = set(["agent1"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 0)
        self.assertEqual(len(g.edges), 0)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {})

    def test_graph6(self):
        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent2", 4: "agent4", 5: "agent2", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_node("agent4"))
        self.assertTrue(g.has_edge("agent4", "agent1"))
        self.assertEqual(g.get_edge_data("agent4", "agent1")[0]["client"], 1)
        self.assertEqual(g.get_edge_data("agent2", "agent4")[0]["client"], 5)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {})

    def test_graph7(self):
        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent2", 4: "agent4", 5: "agent2", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent3", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(len(g.edges), 3)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_node("agent3"))
        self.assertTrue(g.has_node("agent4"))
        self.assertEqual(g.get_edge_data("agent4", "agent1")[0]["client"], 1)
        self.assertEqual(g.get_edge_data("agent2", "agent4")[0]["client"], 5)
        self.assertEqual(g.get_edge_data("agent2", "agent3")[0]["client"], 3)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {1: "agent4", 3: "agent2", 5: "agent2"})

    def test_graph8(self):
        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent2", 4: "agent4", 5: "agent2", 6: "agent1"}
        positive_responses = set(["agent1", "agent2", "agent3", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(len(g.edges), 4)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_node("agent3"))
        self.assertTrue(g.has_node("agent4"))
        self.assertEqual(g.get_edge_data("agent4", "agent1")[0]["client"], 1)
        self.assertEqual(g.get_edge_data("agent2", "agent4")[0]["client"], 5)
        self.assertEqual(g.get_edge_data("agent2", "agent3")[0]["client"], 3)
        self.assertEqual(g.get_edge_data("agent1", "agent4")[0]["client"], 6)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {1: "agent4", 3: "agent2", 5: "agent2", 6: "agent1"})

    def test_graph9(self):

        outcome = {0: "agent1", 1: "agent4", 2: "agent2", 3: "agent2", 4: "agent4", 5: "agent2", 6: "agent1"}
        positive_responses = set(["agent2", "agent3", "agent4"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)
        self.assertTrue(g.has_node("agent2"))
        self.assertTrue(g.has_node("agent3"))
        self.assertTrue(g.has_node("agent4"))
        self.assertEqual(g.get_edge_data("agent2", "agent4")[0]["client"], 5)
        self.assertEqual(g.get_edge_data("agent2", "agent3")[0]["client"], 3)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {})

    def test_graph10(self):

        outcome = {0: "agent2", 1: "agent1", 2: "agent1", 3: "agent4", 4: "agent3", 5: "agent4", 6: "agent4"}
        positive_responses = set(["agent1", "agent2", "agent3"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 2)
        self.assertTrue(g.has_node("agent1"))
        self.assertTrue(g.has_node("agent2"))

        self.assertEqual(g.get_edge_data("agent2", "agent1")[0]["client"], 0)
        self.assertEqual(g.get_edge_data("agent1", "agent2")[0]["client"], 2)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {0: "agent2", 2: "agent1"})

    def test_graph11(self):

        outcome = {0: "agent2", 1: "agent1", 2: "agent1", 3: "agent4", 4: "agent3", 5: "agent4", 6: "agent4"}
        positive_responses = set(["agent1", "agent3"])
        g = self.smtp.build_graph(outcome, positive_responses)

        self.assertEqual(len(g.nodes), 0)
        self.assertEqual(len(g.edges), 0)

        ac = self.smtp.get_acceptable_part(outcome, g)
        self.assertEqual(ac, {})






    #def test_add_agent(self):
    #    agent1 = BasicDARPNegotiator("agent1", BasicDARPProblem(1))
    #    agent2 = BasicDARPNegotiator("agent2", BasicDARPProblem(2))
    #    agent3 = BasicDARPNegotiator("agent3", BasicDARPProblem(3))
    #    agent4 = BasicDARPNegotiator("agent4", BasicDARPProblem(4))

    #    self.smtp.add_participant(agent1)
    #    self.smtp.add_participant(agent2)
    #    self.smtp.add_participant(agent3)
    #    self.smtp.add_participant(agent4)

    #    self.smtp.remove_participant("agent1")
    #    self.assertEqual(len(self.smtp.participants), 3)
    #    self.assertFalse(agent1 in self.smtp.participants)
    #    self.assertTrue(agent2 in self.smtp.participants)
    #    self.assertTrue(agent3 in self.smtp.participants)
    #    self.assertTrue(agent4 in self.smtp.participants)

    #    self.assertRaises(Exception, lambda: self.smtp.remove_participant("agent5"))
