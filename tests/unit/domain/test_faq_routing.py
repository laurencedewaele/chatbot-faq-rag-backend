import unittest

from src.domain.errors.exceptions import InvalidThresholdsError
from src.domain.model.faq_models import FaqRoute, decide_faq_route


class DecideFaqRouteTests(unittest.TestCase):
    def test_no_candidate_falls_back(self):
        self.assertIs(decide_faq_route(None, 0.9, 0.6), FaqRoute.FALLBACK)

    def test_score_at_accept_threshold_is_accepted(self):
        self.assertIs(decide_faq_route(0.9, 0.9, 0.6), FaqRoute.ACCEPT)

    def test_score_above_accept_threshold_is_accepted(self):
        self.assertIs(decide_faq_route(0.99, 0.9, 0.6), FaqRoute.ACCEPT)

    def test_score_between_thresholds_goes_to_judge(self):
        self.assertIs(decide_faq_route(0.75, 0.9, 0.6), FaqRoute.JUDGE)

    def test_score_at_judge_threshold_goes_to_judge(self):
        self.assertIs(decide_faq_route(0.6, 0.9, 0.6), FaqRoute.JUDGE)

    def test_score_below_judge_threshold_falls_back(self):
        self.assertIs(decide_faq_route(0.59, 0.9, 0.6), FaqRoute.FALLBACK)

    def test_equal_thresholds_disable_the_judge(self):
        self.assertIs(decide_faq_route(0.8, 0.8, 0.8), FaqRoute.ACCEPT)
        self.assertIs(decide_faq_route(0.79, 0.8, 0.8), FaqRoute.FALLBACK)

    def test_inconsistent_thresholds_raise(self):
        with self.assertRaises(InvalidThresholdsError):
            decide_faq_route(0.5, 0.6, 0.9)


if __name__ == "__main__":
    unittest.main()
