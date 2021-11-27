from datetime import datetime
import os
import shutil
import unittest

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedSGD
from context import PytorchModel
from learning_model import Linear,LR
from worker import Worker
from server import ParameterServer
from torch import optim
import torch.nn as nn





class FedSGDTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.001
        self.n_max_rounds = 28
        self.n_max_rounds_1= 11
        self.log_interval = 20
        self.n_round_samples = 1600
        self.testbase = self.TEST_BASE_DIR
        self.n_users = 40
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(Linear().state_dict(), self.init_model_path)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir, resultdir=self.RESULT_DIR)
        self.ps.model_1 = LR()
        self.ps.optimizer_1 = optim.AdamW(self.ps.model_1.parameters(),
                                       lr=1e-4,
                                       weight_decay=1e-5)
        self.workers = []
        for u in range(0, self.n_users):
            self.workers.append(Worker(user_idx=u))

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_federated_SGD(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")

        # let workers preprocess data
        for u in range(0, self.n_users):
            self.workers[u].preprocess_worker_data()

        training_start = datetime.now()

        self.ps.model = Linear()
        self.ps.optimizer = optim.AdamW(self.ps.model.parameters(),
                                lr=1e-3,
                                weight_decay=5e-4)
        for r in range(1, self.n_max_rounds + 1):
            path = '/tmp/competetion-test/model.pth'

            start = datetime.now()
            sum_f1 = 0.
            for u in range(0, self.n_users):
                # model.load_state_dict(torch.load(path))
                grads, worker_info, self.ps.model, f1, self.ps.optimizer = self.workers[u].user_round_train(
                    model=self.ps.model, device=device,
                    n_round=r,
                    batch_size=self.batch_size,
                    n_round_samples=self.n_round_samples,
                    optimizer=self.ps.optimizer)

                self.ps.receive_grads_info(grads=grads)
                self.ps.receive_worker_info(
                worker_info)  # The transfer of information from the worker to the server requires a call to the "ps.receive_worker_info"
                self.ps.process_round_train_acc()
                sum_f1 += f1
            self.ps.aggregate()
            print('\nRound {} cost: {}, total training cost: {}, total training f1: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
                sum_f1 / self.n_users,
            ))
        for i in range(0, self.n_users):
            self.workers[i].createbyedges(model=self.ps.model)
        for r in range(1, self.n_max_rounds_1 + 1):
            start = datetime.now()
            sum_f1 = 0.
            for u in range(0, self.n_users):
                f1, self.ps.model_1, self.ps.optimizer_1= self.workers[u].trainbyedges(model_1=self.ps.model_1,optimizer_1=self.ps.optimizer_1)
                sum_f1 += f1
            print('\nRound {} cost: {}, total training cost: {}, total training f1: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
                sum_f1 / self.n_users,
            ))
        if self.ps.model is not None:
            self.ps.save_testdata_prediction(model=self.ps.model, model_1=self.ps.model_1, device=device,
                                             test_batch_size=self.test_batch_size)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedSGDTestSuit('test_federated_SGD'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()
