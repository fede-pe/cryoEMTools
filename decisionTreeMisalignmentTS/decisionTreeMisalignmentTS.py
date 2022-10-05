"""
/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""

import sys
import random

import matplotlib.pyplot as plt
from sklearn import tree, metrics
import numpy as np

TEST_SPLIT = 0.2


class ScriptTomoDecisionTree:

    feature_names = ['averageFiducialResidualsInImage_0',
                     'averageFiducialResidualsInImage_0.5',
                     'averageFiducialResidualsInImage_1',
                     'stdFiducialResidualsInImage_0',
                     'stdFiducialResidualsInImage_0.5',
                     'stdFiducialResidualsInImage_1',
                     'averageResidualDistancePerFiducial_0',
                     'averageResidualDistancePerFiducial_0.5',
                     'averageResidualDistancePerFiducial_1',
                     'stdResidualDistancePerFiducial_0',
                     'stdResidualDistancePerFiducial_0.5',
                     'stdResidualDistancePerFiducial_1',
                     'ratioOfImagesOutOfRange_0',
                     'ratioOfImagesOutOfRange_0.5',
                     'ratioOfImagesOutOfRange_1',
                     'longestMisalignedChain_0',
                     'longestMisalignedChain_0.5',
                     'longestMisalignedChain_1']

    def __init__(self, filePath):
        self.filePath = filePath

        self.dtc = tree.DecisionTreeClassifier()

        self.testSplit = 0.2

        self.infoData_train = []
        self.classData_train = []
        self.infoData_test = []
        self.classData_test = []

        self.readInputData()

        self.trainDecisionTree()

        self.testDecisionTree()

    def readInputData(self):
        """
          Read input data
        """

        print("Reading input data from " + self.filePath)

        X = []
        y = []

        with open(self.filePath) as f:
            lines = f.readlines()

            random.shuffle(lines)

            for line in lines:
                vector = line.split(',')
                vector_f = [float(i) for i in vector]
                X.append(vector_f[:-1])
                y.append(vector_f[-1])

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        X_array.reshape(-1, 1)
        y_array.reshape(-1, 1)

        testSize = int(TEST_SPLIT * len(X))

        self.infoData_train = X_array[:testSize]
        self.classData_train = y_array[:testSize]
        self.infoData_test = X_array[testSize:]
        self.classData_test = y_array[testSize:]

    def trainDecisionTree(self):
        """
          Train decision tree
        """

        print("Training tree")

        self.dtc.fit(self.infoData_train, self.classData_train)

        text_representation = tree.export_text(self.dtc)
        print(text_representation)

        tree.plot_tree(self.dtc,
                       feature_names=self.feature_names,
                       filled=True)
        plt.show()

    def testDecisionTree(self):
        """
          Test decision tree
        """

        print("Testing tree")

        y_predict = self.dtc.predict(self.infoData_test)

        print("Model accuracy: " + str(metrics.accuracy_score(self.classData_test, y_predict)))


if __name__ == '__main__':
    # Check no program arguments missing
    if len(sys.argv) != 2:
        print("Usage: python decisionTreeMisalignmentTS.py <infoFilePath> ")
        sys.exit()

    # Path with the input stack of data
    filePath = sys.argv[1]

    cdt = ScriptTomoDecisionTree(filePath)

