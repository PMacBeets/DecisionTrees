
import itertools
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree
import numpy as np

class Gob(object):
    pass

class BinaryTruthTable(object):
    # Extension of Code BY..
    #
    # Copyright 2016 Trey Morris
    #
    #   Licensed under the Apache License, Version 2.0 (the "License");
    #   you may not use this file except in compliance with the License.
    #   You may obtain a copy of the License at
    #
    #       http://www.apache.org/licenses/LICENSE-2.0
    #
    #   Unless required by applicable law or agreed to in writing, software
    #   distributed under the License is distributed on an "AS IS" BASIS,
    #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    #   See the License for the specific language governing permissions and
    #   limitations under the License.

    def __init__(self, state_machines, phrases, ints=True):

        self.base = [state.name for state in state_machines]
        self.phrases = phrases or []
        self.ints= ints

        # generate the sets of booleans for the bases
        self.base_conditions = list(itertools.product([False, True],
                                                      repeat=len(self.base)))

        # regex to match whole words defined in self.bases
        # used to add object context to variables in self.phrases
        self.p = re.compile(r'(?<!\w)(' + '|'.join(self.base) + ')(?!\w)')

        arr = np.zeros((len(self.base_conditions), len(self.base + self.phrases)))

        i = 0
        for conditions_set in self.base_conditions:
            arr[i, :] = self.calculate(*conditions_set)
            i += 1
            # self.df.loc[self.calculate(*conditions_set)]

        self.df = pd.DataFrame(arr, columns=self.base + self.phrases)
        super().__init__(state_machines, self.df[self.base], self.df[self.phrases])

    def calculate(self, *args):
        # store bases in an object context
        g = Gob()
        for a, b in zip(self.base, args):
            setattr(g, a, b)

        # add object context to any base variables in self.phrases
        # then evaluate each
        eval_phrases = []
        for item in self.phrases:
            item = self.p.sub(r'g.\1', item)
            eval_phrases.append(eval(item))

        # add the bases and evaluated phrases to create a single row
        row = [getattr(g, b) for b in self.base] + eval_phrases
        if self.ints:
            return [int(c) for c in row]
        else:
            return row
