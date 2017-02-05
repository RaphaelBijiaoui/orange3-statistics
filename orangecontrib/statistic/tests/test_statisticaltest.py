import unittest

from statistic.widgets.statisticaltest import FisherTest


class StatisticalTestTest(unittest.TestCase):
    def test_fisher_test_params(self):
        self.assertFalse(FisherTest.has_one_sample,
                         'Fisher test can\'t be applied to only one sample!')
        self.assertTrue(FisherTest.has_two_sample,
                        'Fisher test must can be applied to two samples!')
        self.assertFalse(
            FisherTest.has_many_sample,
            'Fisher test can\'t be applied to more than two samples!')

        self.assertFlase(FisherTest.use_continuous_data,
                         'Fisher test can\'t be applied to continuous data')
        self.assertTrue(FisherTest.use_discrete_data,
                        'Fisher test must can be applied to discrete data')
