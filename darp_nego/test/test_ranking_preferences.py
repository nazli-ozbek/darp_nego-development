import unittest
from darp_nego.core.utils import RankingPreferences

class TestRankingPreferences(unittest.TestCase):

    def setUp(self) -> None:

        self.ranking1 = RankingPreferences()
        for i in range(7):
            self.ranking1.add_client(i)

        self.ranking1.update_prefers(0, 1)
        self.ranking1.update_prefers(3, 4)
        self.ranking1.update_prefers(1, 4)
        self.ranking1.update_prefers(4, 6)
        self.ranking1.update_prefers(2, 1)

    def test_preferences_1(self):
        prefers = self.ranking1.get_better_assignment(1)
        self.assertSetEqual({4, 6}, prefers)
        worse = self.ranking1.get_worse_assignments(4)
        self.assertSetEqual({0, 1, 2, 3}, worse)
        self.assertAlmostEqual(0.5, self.ranking1.similarity_full_ranking(1))
        self.assertAlmostEqual((0.5+0.75)/2, self.ranking1.simulate_similarity_full_ranking(1, 5))
        self.assertAlmostEqual(0.5, self.ranking1.simulate_similarity_full_ranking(4, 6))
        self.assertAlmostEqual(0.5, self.ranking1.simulate_similarity_full_ranking(4, 6))
        self.assertAlmostEqual(0.75, self.ranking1.simulate_similarity_full_ranking(1, 3))
        self.assertAlmostEqual(0.5, self.ranking1.simulate_similarity_full_ranking(3, 5))
        self.assertAlmostEqual(0.5, self.ranking1.simulate_similarity_full_ranking(3, 6))
        self.assertAlmostEqual((4/6 + 0.5)/2, self.ranking1.simulate_similarity_full_ranking(0, 5))

        for i in range(7):
            for j in range(7):
                if i!=j:
                    print(i, j, self.ranking1.simulate_similarity_full_ranking(i, j))



