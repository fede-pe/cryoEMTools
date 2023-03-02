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
from past.builtins import raw_input
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import numpy as np

TEST_SPLIT = 0.2

FEATURE_NAMES_CHAIN = ['avgResidual',
                       'stdResidual',
                       'chArea',
                       'chPerimeter',
                       'pvBinX',
                       'pvBinY',
                       'pvF',
                       'pvADF',
                       'imagesOutOfRange',
                       'LongestMisaliChain']

FEATURE_NAMES_IMAGE = ['avgResidual',
                       'stdResidual',
                       'chArea',
                       'chPerimeter',
                       'pvBinX',
                       'pvBinY',
                       'pvF',
                       'markerOutOfRange']


class ScriptTomoDecisionTree:
    def __init__(self, filePath, treeMode, trainMode):
        self.filePath = filePath

        self.testSplit = TEST_SPLIT

        self.infoData_train = []
        self.classData_train = []
        self.infoData_test = []
        self.classData_test = []

        if treeMode == "0":  # Tree training
            print("Tree mode")
            self.dtc = tree.DecisionTreeClassifier(random_state=0)
        else:  # Forest training
            print("Forest mode")
            self.dtc = RandomForestClassifier(max_depth=2,
                                              random_state=0)

        if trainMode == "0":  # Chain mode
            print("Training chain mode")
            self.feature_names = FEATURE_NAMES_CHAIN
        else:  # Image mode
            print("Training image mode")
            self.feature_names = FEATURE_NAMES_IMAGE

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

        self.infoData_train = X_array[testSize:]
        self.classData_train = y_array[testSize:]
        self.infoData_test = X_array[:testSize]
        self.classData_test = y_array[: testSize]

    def trainDecisionTree(self):
        """
          Train decision tree
        """

        print("Training...")

        self.dtc.fit(self.infoData_train, self.classData_train)

        if treeMode == "0":  # Tree training
            # Pruning
            prun = self.dtc.cost_complexity_pruning_path(self.infoData_train, self.classData_train)
            alphas = prun['ccp_alphas']

            clfs = []
            for alpha in alphas:
                clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
                clf.fit(self.infoData_train, self.classData_train)

                clfs.append(clf)

            clfs = clfs[:-1]
            ccp_alphas = alphas[:-1]
            node_counts = [clf.tree_.node_count for clf in clfs]
            depth = [clf.tree_.max_depth for clf in clfs]
            plt.scatter(ccp_alphas, node_counts)
            plt.scatter(ccp_alphas, depth)
            plt.plot(ccp_alphas, node_counts, label='no of nodes', drawstyle="steps-post")
            plt.plot(ccp_alphas, depth, label='depth', drawstyle="steps-post")
            plt.legend()
            plt.show()


            train_acc = []
            test_acc = []
            for c in clfs:
                y_train_pred = c.predict(self.infoData_train)
                y_test_pred = c.predict(self.infoData_test)
                train_acc.append(metrics.accuracy_score(y_train_pred, self.classData_train))
                test_acc.append(metrics.accuracy_score(y_test_pred, self.classData_test))

            plt.scatter(ccp_alphas, train_acc)
            plt.scatter(ccp_alphas, test_acc)
            plt.plot(ccp_alphas, train_acc, label='train_accuracy', drawstyle="steps-post")
            plt.plot(ccp_alphas, test_acc, label='test_accuracy', drawstyle="steps-post")
            plt.legend()
            plt.title('Accuracy vs alpha')
            plt.show()

            input_alpha = float(raw_input("Enter alpha:\n"))

            clf_ = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=input_alpha)
            clf_.fit(self.infoData_train, self.classData_train)

            tree.plot_tree(clf_,
                           feature_names=self.feature_names,
                           filled=True,
                           fontsize=12)
            plt.show()

            # No prunning
            # text_representation = tree.export_text(self.dtc)
            # print(text_representation)

            # tree.plot_tree(self.dtc,
            #                feature_names=self.feature_names,
            #                filled=True,
            #                fontsize=12)
            # plt.show()


        else:  # Forest training
            import pandas as pd
            feature_imp = pd.Series(self.dtc.feature_importances_, index=self.feature_names).sort_values(ascending=False)

            # Creating a bar plot
            sns.barplot(x=feature_imp,
                        y=feature_imp.index)
            # Add labels to your graph
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualizing Important Features")
            plt.legend()
            plt.show()

    def testDecisionTree(self):
        """
          Test decision tree
        """

        print("Testing...")

        y_predict = self.dtc.predict(self.infoData_test)

        print("Model accuracy: " + str(metrics.accuracy_score(self.classData_test, y_predict)))


if __name__ == '__main__':
    # Check no program arguments missing
    if len(sys.argv) != 4:
        print("Usage: python decisionTreeMisalignmentTS.py <infoFilePath> <treeMode 0 (tree) /1 (forest)> "
              "<trainMode 0 (chain) / 1 (image)>")
        sys.exit()

    # Path with the input stack of data
    filePath = sys.argv[1]
    treeMode = sys.argv[2]
    trainMode = sys.argv[3]

    if treeMode != "0" and treeMode != "1":
        print(treeMode)
        print("ERROR IN TREE MODE. Usage: python decisionTreeMisalignmentTS.py <infoFilePath> "
              "<treeMode 0 (tree) /1 (forest)> <trainMode 0 (chain) / 1 (image)>")
        sys.exit()

    if trainMode != "0" and trainMode != "1":
        print(treeMode)
        print("ERROR IN TRAIN MODE. Usage: python decisionTreeMisalignmentTS.py <infoFilePath> "
              "<treeMode 0 (tree) /1 (forest)> <trainMode 0 (chain) / 1 (image)>")
        sys.exit()

    cdt = ScriptTomoDecisionTree(filePath, treeMode, trainMode)
