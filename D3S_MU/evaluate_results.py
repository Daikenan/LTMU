import sys
import os
sys.path.append(os.path.join('../utils'))
import neuron.data as data


def evaluate(dataset, trackers=None):
    if dataset == 'lasot':
        evaluators = [data.EvaluatorLaSOT()]
    elif dataset == 'tlp':
        evaluators = [data.EvaluatorTLP()]
    if trackers is None:
        trackers = os.listdir('./results')
    for e in evaluators:
        e.report(trackers, dataset=dataset, plot_curves=True)


if __name__ == '__main__':
    evaluate(dataset='lasot')
    evaluate(dataset='tlp')


