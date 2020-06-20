import unittest

from neuron.models.trackers import DummyTracker, DummyOxUvA_Tracker
from neuron.data.evaluators import *


class TestSeqEvaluators(unittest.TestCase):

    def setUp(self):
        self.tracker = DummyTracker()
        self.oxuva_tracker = DummyOxUvA_Tracker()
        self.visualize = False

    def test_otb_eval(self):
        evaluators = [
            EvaluatorOTB(version=2013),
            EvaluatorOTB(version=2015),
            EvaluatorDTB70(),
            EvaluatorNfS(fps=30),
            EvaluatorNfS(fps=240),
            EvaluatorTColor128(),
            EvaluatorUAV123(version='UAV123'),
            EvaluatorUAV123(version='UAV20L'),
            EvaluatorLaSOT(),
            EvaluatorVisDrone(subset='val')]
        for e in evaluators:
            e.run(self.tracker, visualize=self.visualize)
            e.report(self.tracker.name)

    def test_vot_eval(self):
        evaluators = [
            EvaluatorVOT(version=2018),
            EvaluatorVOT(version=2019)]
        for e in evaluators:
            e.run(self.tracker, visualize=self.visualize)
            e.report(self.tracker.name)
    
    def test_got10k_eval(self):
        evaluators = [
            EvaluatorGOT10k(subset='val'),
            EvaluatorGOT10k(subset='test')]
        for e in evaluators:
            e.run(self.tracker, visualize=self.visualize)
            e.report(self.tracker.name)
    
    def test_oxuva_eval(self):
        evaluators = [
            EvaluatorOxUvA(subset='dev'),
            EvaluatorOxUvA(subset='test')]
        for e in evaluators:
            e.run(self.oxuva_tracker, visualize=self.visualize)


if __name__ == '__main__':
    unittest.main()
