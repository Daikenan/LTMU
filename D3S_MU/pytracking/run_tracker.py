import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.tpldataset import TPLDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.vot18dataset import VOT18Dataset
from pytracking.evaluation.lasotdataset import LaSOTDataset
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset
from pytracking.evaluation.got10kdataset import GOT10KDatasetTest, GOT10KDatasetVal, GOT10KDatasetLTRVal
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    if dataset_name == 'otb':
        dataset = OTBDataset()
    elif dataset_name == 'nfs':
        dataset = NFSDataset()
    elif dataset_name == 'uav':
        dataset = UAVDataset()
    elif dataset_name == 'tpl':
        dataset = TPLDataset()
    elif dataset_name == 'vot':
        dataset = VOTDataset()
    elif dataset_name == 'vot18':
        dataset = VOT18Dataset()
    elif dataset_name == 'otb_vot':
        dataset = OTB100Dataset()
    elif dataset_name == 'tn':
        dataset = TrackingNetDataset()
    elif dataset_name == 'gott':
        dataset = GOT10KDatasetTest()
    elif dataset_name == 'gotv':
        dataset = GOT10KDatasetVal()
    elif dataset_name == 'gotlv':
        dataset = GOT10KDatasetLTRVal()
    elif dataset_name == 'lasot':
        dataset = LaSOTDataset()
    else:
        raise ValueError('Unknown dataset name')

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_dataset(dataset, trackers, debug, threads)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')

    args = parser.parse_args()

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset, args.sequence, args.debug, args.threads)


if __name__ == '__main__':
    main()
