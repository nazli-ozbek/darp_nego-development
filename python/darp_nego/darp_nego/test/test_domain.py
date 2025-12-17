import unittest
from darp_nego.outcome import DARPNegotiationDomain
from darp_nego.darp import BasicDARPClient

class TestBasicDARPDomain(unittest.TestCase):

    def setUp(self) -> None:
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

    def test_add_domain(self):
        self.assertEqual(self.domain1.get_current_owner(0), "agent1")
        self.assertEqual(self.domain1.get_current_owner(1), "agent1")
        self.assertEqual(self.domain1.get_current_owner(2), "agent2")
        self.assertEqual(self.domain1.get_current_owner(3), "agent3")
        self.assertEqual(self.domain1.get_current_owner(4), "agent4")
        self.assertEqual(self.domain1.get_current_owner(5), "agent4")
        self.assertEqual(self.domain1.get_current_owner(6), "agent4")

    def test_update_client(self):
        self.domain1.update_client(1, "agent2")
        self.assertEqual(self.domain1.get_current_owner(1), "agent2")

    def test_get_client(self):
        self.assertEqual(self.domain1.get_client(4), self.client5)
        self.assertRaises(Exception, lambda : self.domain1.get_client(8))

    def test_generate_random_outcome(self):
        outcome = self.domain1.generate_random_outcome()
        owners = list(outcome.values())
        clients = list(outcome.keys())
        self.assertEqual(len(clients), 7)
        for i in range(7):
            self.assertTrue(i in clients)
        for owner in owners:
            self.assertTrue(owner in ["agent1", "agent2", "agent3", "agent4"])

    def test_involves1(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent2", 3: "agent3", 4: "agent4", 5:"agent4", 6:"agent4"}
        involves = self.domain1.get_involved_agents(outcome)
        self.assertEqual(len(involves), 2)
        self.assertTrue("agent1" in involves)
        self.assertTrue("agent2" in involves)

    def test_involves2(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent3", 3: "agent3", 4: "agent4", 5: "agent4", 6: "agent4"}
        involves = self.domain1.get_involved_agents(outcome)
        self.assertEqual(len(involves), 3)
        self.assertTrue("agent1" in involves)
        self.assertTrue("agent2" in involves)
        self.assertTrue("agent3" in involves)

    def test_involves3(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent3", 3: "agent3", 4: "agent4", 5: "agent2", 6: "agent4"}
        involves = self.domain1.get_involved_agents(outcome)
        self.assertEqual(len(involves), 4)
        self.assertTrue("agent1" in involves)
        self.assertTrue("agent2" in involves)
        self.assertTrue("agent3" in involves)
        self.assertTrue("agent4" in involves)

    def test_gets1(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent2", 3: "agent3", 4: "agent4", 5: "agent4", 6: "agent4"}
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent1"), [])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent2"), [1])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent3"), [])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent4"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent1"), [1])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent2"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent3"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent4"), [])

    def test_gets2(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent3", 3: "agent3", 4: "agent4", 5: "agent4", 6: "agent4"}
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent1"), [])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent2"), [1])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent3"), [2])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent4"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent1"), [1])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent2"), [2])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent3"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent4"), [])

    def test_gets3(self):
        outcome = {0: "agent1", 1: "agent2", 2: "agent3", 3: "agent3", 4: "agent4", 5: "agent2", 6: "agent4"}
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent1"), [])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent2"), [1,5])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent3"), [2])
        self.assertListEqual(self.domain1.gets_clients(outcome, "agent4"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent1"), [1])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent2"), [2])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent3"), [])
        self.assertListEqual(self.domain1.gives_clients(outcome, "agent4"), [5])

    def test_generate_outcome_cluster(self):
        """Test the new generate_outcome_cluster function"""
        # Create test domain with some clients
        clients = [BasicDARPClient(i, (0, 0), (1, 1), 0, 100, 0, 100, 1) for i in range(4)]
        owners = ["agent1", "agent2", "agent1", "agent2"]
        domain = DARPNegotiationDomain(clients, owners)
        
        # Create mock clusters
        clusters = {
            0: [(0, clients[0]), (1, clients[1])],  # Cluster 0: clients 0, 1
            1: [(2, clients[2]), (3, clients[3])]   # Cluster 1: clients 2, 3
        }
        
        # Create mock round history
        round_history = [
            {
                "proposed_outcome": {0: "agent2", 1: "agent1"},
                "agent_responses": {
                    "agent1": {"accepted": True},
                    "agent2": {"accepted": False}
                }
            }
        ]
        
        # Test the function
        outcome = domain.generate_outcome_cluster(clusters, round_history, max_swaps_per_pair=1)
        
        # Verify the outcome structure
        self.assertIsInstance(outcome, dict)
        self.assertEqual(len(outcome), 4)  # Should have 4 clients
        
        # Verify all clients are assigned to agents
        for client_id in range(4):
            self.assertIn(client_id, outcome)
            self.assertIn(outcome[client_id], ["agent1", "agent2"])
        
        # Verify the outcome is different from current ownership (some swaps should happen)
        current_owners = {0: "agent1", 1: "agent2", 2: "agent1", 3: "agent2"}
        has_changes = any(outcome[client_id] != current_owners[client_id] for client_id in range(4))
        self.assertTrue(has_changes, "Expected some client swaps to occur")


if __name__ == '__main__':
    unittest.main()
