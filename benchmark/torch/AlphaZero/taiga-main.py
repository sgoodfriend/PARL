#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Coach import Coach
from connect4_game import Connect4Game
from utils import *

from parl.utils import logger

load_folder_file = (
    "../../../../connect_x/runs/taiga_08_06",
    "best.pth.tar",
)

args = dotdict({
    # master address of xparl cluster
    "master_address": "localhost:8010",
    # number of remote actors (execute tasks [self-play/pitting/evaluate_test_dataset] in parallel).
    "actors_num": 10,

    # total number of iteration
    "numIters": 50,
    # Number of complete self-play games to simulate during a new iteration.
    "numEps": 500,
    # Number of games to play during arena (pitting) play to determine if new neural network will be accepted.
    "arenaCompare": 60,
    # Number of games moves for MCTS to simulate.
    "numMCTSSims": 800,
    # temp=1 (Temperature, τ (tau)) if episodeStep < tempThresholdStep, and thereafter uses temp=0.
    "tempThresholdStep": 15,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    "updateThreshold": 0.6,
    # CPUCT parameter
    "cpuct": 4,
    # alpha parameter of dirichlet noise which is added to the policy (pi)
    "dirichletAlpha": 0.48,
    # history of examples from numItersForTrainExamplesHistory latest iterations (training data)
    "numItersForTrainExamplesHistory": 100,

    # folder to save model and training examples
    "checkpoint": "./saved_model/",
    # whether to load saved model and training examples
    "load_model": True,
    "load_folder_file": load_folder_file,
})

# Plays arenaCompare games in which player1 starts arenaCompare/2 games and player2 starts arenaCompare/2 games.
assert args.arenaCompare % 2 == 0

# make sure the tasks can be split evenly among different remote actors
assert args.numEps % args.actors_num == 0
assert (args.arenaCompare // 2) % args.actors_num == 0
assert 1000 % args.actors_num == 0  # there are 1000 boards state in test_dataset


def main():
    game = Connect4Game()

    c = Coach(game, args)

    if args.load_model:
        logger.info("Loading checkpoint {}...".format(args.load_folder_file))
        c.loadModel()
        logger.info("Loading 'trainExamples' from file {}...".format(
            args.load_folder_file))
        c.loadTrainExamples()

    c.learn()


if __name__ == "__main__":
    main()
