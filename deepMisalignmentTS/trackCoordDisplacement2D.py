
import numpy as np
import scipy.stats
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

import errno
import os
import sys
import math
import csv
import random
import glob


class TrackerDisplacement:
    def __init__(self, pathCoordinate3DFolder,
                 pathAnglesFolder,
                 pathMisalignmentMatrixFolder,
                 pathPatternToSubtomoFilesFolder):

        self.generateOutputHistogramPlots = False
        self.generateOutputHullPlot = False

        self.getXYCoordinatesMatrix = [[1, 0, 0],
                                       [0, 1, 0]]

        # self.maximumOrderMoment = 7

        # Iterate for all the set of coordinates
        pathCoordinate3DRegex = os.path.join(pathCoordinate3DFolder + "*.xmd")

        print("Metadata processed:\n")

        for file in glob.glob(pathCoordinate3DRegex):

            print(file)

            fileName = os.path.splitext(os.path.basename(file))[0]

            # Generate paths to input data
            pathCoordinate3D = file
            pathAngles = os.path.join(pathAnglesFolder, fileName + ".tlt")
            pathMisalignmentMatrix = os.path.join(pathMisalignmentMatrixFolder, fileName, "TM_" + fileName + ".xf")
            pathPatternToSubtomoFiles = os.path.join(pathPatternToSubtomoFilesFolder, fileName + "*.mrc")

            # Check that files exist
            if not os.path.exists(pathCoordinate3D):
                raise Exception("Path to coordinate 3d files %s does not exist." % pathCoordinate3D)

            if not os.path.exists(pathAngles):
                raise Exception("Path to angle file %s does not exist." % pathAngles)

            if not os.path.exists(pathMisalignmentMatrix):
                raise Exception("Path to misalignment matrix %s does not exist." % pathMisalignmentMatrix)

            subtomos = self.getSubtomoList(pathPatternToSubtomoFiles)
            coordinates3D = self.readCoordinates3D(pathCoordinate3D)
            angles = self.readAngleFile(pathAngles)

            if len(subtomos) != len(coordinates3D):
                raise Exception("Subtomo list and coordinate list length must be equal.\n"
                                "Subtomo list length: %d\n"
                                "Coordinate list length: %d\n" % (len(subtomos), len(coordinates3D)))

            misalignmentMatrices = self.readMisalignmentMatrix(pathMisalignmentMatrix)

            vectorDistance2D = []
            vectorMisalignment2D = []

            for coordinate3D, subtomo in zip(coordinates3D, subtomos):
                for indexAngle, angle in enumerate(angles):
                    projectedCoordinate2D = self.getProjectedCoordinate2D(angle,
                                                                          coordinate3D)

                    misalignedProjectedCoordinate2D = \
                        self.getMisalignedProjectedCoordinate2D(angle,
                                                                coordinate3D,
                                                                misalignmentMatrices[:, :, indexAngle])

                    vectorMisalignment2D.append(self.getMisalignmentVector(projectedCoordinate2D,
                                                                           misalignedProjectedCoordinate2D))

                    vectorDistance2D.append(self.getDistance2D(projectedCoordinate2D,
                                                               misalignedProjectedCoordinate2D))

                maximumDistance = self.getMaximumDistance(vectorDistance2D)
                totalDistance = self.getTotalDistance(vectorDistance2D)

                hull = self.getConvexHull(vectorMisalignment2D)

                hullArea = [hull.area]
                hullPerimeter = self.getHullPerimeter(hull)

                pca = self.getPCA(vectorMisalignment2D)[0]

                statistics = maximumDistance + totalDistance + hullArea + hullPerimeter + [pca[0]] + [pca[1]]

                self.saveStaticts(statistics + [subtomo])

                vectorDistance2D = []

            self.createSubtomoLinks(subtomos)

    def getProjectedCoordinate2D(self, angle, coordinate3D):
        """ Method to calculate the projection of a 3D coordinate onto a plane defined by its angle (rotates
        around the Y axis). """

        rotationMatrix = self.getRotationMatrix(angle)

        projectedCoordinate2D = np.matmul(self.getXYCoordinatesMatrix,
                                          np.matmul(rotationMatrix,
                                                    coordinate3D))

        return projectedCoordinate2D

    def getMisalignedProjectedCoordinate2D(self, angle, coordinate3D, misalignmentMatrix):
        """ Method to calculate the projection of a 3D coordinate onto a plane defined by its angle (rotates
        around the Y axis) considering the misalginment introduce in its direction. """

        projectedCoordiante2D = self.getProjectedCoordinate2D(angle, coordinate3D)

        misalignedProjectedCoordinate2D = np.matmul(misalignmentMatrix,
                                                    [projectedCoordiante2D[0],
                                                     projectedCoordiante2D[1],
                                                     1])

        return [misalignedProjectedCoordinate2D[0], misalignedProjectedCoordinate2D[1]]

    @staticmethod
    def getMaximumDistance(distanceVector):
        """ Method to calculate the maximum distance in trajectory """

        return ['%.4f' % max(distanceVector)]

    @staticmethod
    def getTotalDistance(distanceVector):
        """ Method to calculate the total distance of the trajectory """

        totalDistance = 0

        for distance in distanceVector:
            totalDistance += distance

        return ['%.4f' % totalDistance]

    def getConvexHull(self, vectorMisalignment2D):
        """ Method to calculate the convex hull containing the misalignment coordinates """

        convexHull = ConvexHull(vectorMisalignment2D)

        if self.generateOutputHullPlot:
            import matplotlib.pyplot as plt
            plt.scatter(*zip(*convexHull.points))
            plt.scatter(*zip(*convexHull.points[convexHull.vertices]))
            plt.show()

        return convexHull

    def getHullPerimeter(self, hull):
        """ Method to calculate the perimeter occupied by the convex hull in which set of coordinates that describes the
        misalignment introduced for each 3D coordinate at each projection through the tilt-series are contained. """

        perimeter = 0

        hullVertices = []

        for position in hull.vertices:
            hullVertices.append(hull.points[position])

        for i in range(len(hullVertices)):
            shiftedIndex = (i + 1) % len(hullVertices)
            perimeter += self.getDistance2D(np.array(hullVertices[i]), np.array(hullVertices[shiftedIndex]))

        return ['%.4f' % perimeter]

    @staticmethod
    def getPCA(vectorMisalignment2D):
        """ Method to calculate the PCA of the misalignment introduced for each 3D coordinate at each projection
        through the tilt-series distribution in order to characterize the occupied region. """

        pca = PCA(n_components=2)
        pca.fit(vectorMisalignment2D)

        # Return only the first component (remove redundant information)
        return pca.components_

    # ----------------------------------- Utils methods -----------------------------------

    @staticmethod
    def getRotationMatrix(angle):
        """ Method to calculate the 3D rotation matrix of a plane given its tilt angle. """

        angleRad = np.deg2rad(angle)

        rotationMatrix = [[np.cos(angleRad), 0, np.sin(angleRad)],
                          [0, 1, 0],
                          [-np.sin(angleRad), 0, np.cos(angleRad)]]

        return rotationMatrix

    @staticmethod
    def getMisalignmentVector(coordinate2D, misalignedCoordiante2D):
        """ Method to calculate the vector described by the projected 2D coordinate and its misaligned
        correspondent. """

        return [misalignedCoordiante2D[0] - coordinate2D[0], misalignedCoordiante2D[1] - coordinate2D[1]]

    @staticmethod
    def getDistance2D(coordinate2D, misalignedCoordiante2D):
        """ Method to calculate the distance between two 2D coordinates. """

        distanceVector = coordinate2D - misalignedCoordiante2D
        distanceVector = [i ** 2 for i in distanceVector]
        distance = np.sqrt(sum(distanceVector))

        return distance

    @staticmethod
    def getFreedmanDiaconisBinWidth(distanceVector):
        """ Method to calculate the optimal bin width based on the Freedman-Diaconis method. """

        n = len(distanceVector)
        iqr = scipy.stats.iqr(distanceVector)

        binWidth = int(math.floor(2 * (iqr / n ** (1 / 3))) + 1)

        return binWidth

    @staticmethod
    def getSquareRootBinWidth(distanceVector):
        """ Method to calculate the optimal bin width based on the square-root method. """

        n = len(distanceVector)
        binWidth = int(math.floor(math.sqrt(n) + 1))

        return binWidth

    @staticmethod
    def getSturguesBinWidth(distanceVector):
        """ Method to calculate the optimal bin width based on the Sturgues method. It assumes a normal
        distribution. """

        n = len(distanceVector)
        binWidth = int(math.floor(math.log2(n)) + 1)

        return binWidth

    @staticmethod
    def getRiceBinWidth(distanceVector):
        """ Method to calculate the optimal bin width based on the Rice method. It assumes a normal distribution. """

        n = len(distanceVector)
        binWidth = int(math.floor(2 * n ** (1 / 3)) + 1)

        return binWidth

    # ----------------------------------- I/O methods -----------------------------------

    @staticmethod
    def getSubtomoList(pathPatternToSubtomoFiles):
        """ Method to get the list of subtomos peaked from the input coordinates using a regular expression indicating
        its location"""

        subtomoList = []

        for file in glob.glob(pathPatternToSubtomoFiles):
            subtomoList.append(file)

        subtomoList.sort()

        return subtomoList

    @staticmethod
    def readCoordinates3D(filePath):
        """ Method to read 3D coordinate files in Xmipp format (xmd). """

        coordinates = []

        with open(filePath) as f:
            lines = f.readlines()
            lines = lines[7:]
            for line in lines:
                vector = line.split()
                coordinate = ([float(vector[0]),
                               float(vector[1]),
                               float(vector[2])])
                coordinates.append(coordinate)

        return coordinates

    # Read IMOD format coordinates file
    # coordinates = []
    # with open(filePath) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         vector = line.split()
    #         coordinate = ([float(vector[1]),
    #                        float(vector[2]),
    #                        float(vector[3])])
    #         coordinates.append(coordinate)
    #
    # return coordinates

    @staticmethod
    def readAngleFile(filePath):
        """ Method to read angles in .tlt format. """

        angles = []
        with open(filePath) as f:
            lines = f.readlines()
            for line in lines:
                vector = line.split()
                angles.append(float(vector[0]))

        return angles

    @staticmethod
    def readMisalignmentMatrix(filePath):
        """ Method to read the transformation matrix (IMOD format) and returns a 3D matrix containing the
        transformation matrices for each tilt-image belonging to the tilt-series. """

        with open(filePath, "r") as matrix:
            lines = matrix.readlines()
        numberLines = len(lines)
        frameMatrix = np.empty([3, 3, numberLines])
        i = 0
        for line in lines:
            values = line.split()
            frameMatrix[0, 0, i] = float(values[0])
            frameMatrix[1, 0, i] = float(values[2])
            frameMatrix[0, 1, i] = float(values[1])
            frameMatrix[1, 1, i] = float(values[3])
            frameMatrix[0, 2, i] = float(values[4])
            frameMatrix[1, 2, i] = float(values[5])
            frameMatrix[2, 0, i] = 0.0
            frameMatrix[2, 1, i] = 0.0
            frameMatrix[2, 2, i] = 1.0
            i += 1

        return frameMatrix

    @staticmethod
    def saveStaticts(statistics):
        """ Method to save statistics in output file"""

        fieldNames = ['maxDistance', 'totalDistance', 'hullArea', 'hullPerimeter', 'pcaX', 'pcaY', 'subTomoPath']

        # " Create as many fields as moments calculated "
        # for order in range(1, self.maximumOrderMoment + 1):
        #     fieldNames.append('E(X^%d)' % order)

        fileName = 'misalignmentStatistics.txt'
        filePrefix = os.path.dirname(os.path.abspath(sys.argv[0]))
        filePath = os.path.join(filePrefix, 'trainingSet', fileName)

        " Create intermediate directories if missing "
        if not os.path.exists(filePath):
            try:
                os.makedirs(os.path.dirname(filePath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        mode = "a" if os.path.exists(filePath) else "w"

        with open(filePath, mode) as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldNames)

            if mode == "w":
                writer.writeheader()

            writerDict = {
                'maxDistance': statistics[0],
                'totalDistance': statistics[1],
                'hullArea': statistics[2],
                'hullPerimeter': statistics[3],
                'pcaX': statistics[4],
                'pcaY': statistics[5],
                'subTomoPath': statistics[6]
            }

            # for order in range(1, self.maximumOrderMoment + 1):
            #     dicKey = 'E(X^%d)' % order
            #     writerDict[dicKey] = statistics[order + 3]

            # writerDict['subTomoPath'] = statistics[-1]

            writer.writerow(writerDict)

    @staticmethod
    def createSubtomoLinks(subtomos):
        prefix = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'trainingSet')

        # Search ofr the last subtomo saved
        lastIndex = 0
        while True:
            if os.path.exists(os.path.join(prefix, "subtomo%s.mrc" % str(lastIndex).zfill(4))):
                lastIndex += 1
            else:
                break

        for subtomo in subtomos:
            os.symlink(subtomo, os.path.join(prefix, "subtomo%s.mrc" % str(lastIndex).zfill(4)))
            lastIndex += 1

    # ----------------------------------- Unused methods -----------------------------------

    @staticmethod
    def getProjectionMatrix(angle):
        """ Method to calculate the projection matrix of a plane given its tilt angle."""

        angleRad = np.deg2rad(angle)

        # plane matrix
        v1 = [0, 1, 0]
        v2 = [np.cos(angleRad), 0, np.sin(angleRad)]
        v = np.matrix.transpose(np.array([v1, v2]))

        # projection matrix vProj = v (vt v)^-1 vt
        vt = np.matrix.transpose(v)
        vInv = np.linalg.inv(np.matmul(vt, v))

        vProj = np.matmul(v, np.matmul(vInv, vt))

        return vProj

    @staticmethod
    def getReferenceCoordinate(misalignmentVector):
        """ Method to obtain a reference coordinate to sort a set of coordinates in terms of its relative position to
        the reference. The reference is calculated as the average of all the coordinates from the set. """

        n = len(misalignmentVector)
        sumX = 0
        sumY = 0

        for vector in misalignmentVector:
            sumX += vector[0]
            sumY += vector[1]

        referenceCoordinate = [sumX / n, sumY / n]

        return referenceCoordinate

    def getSegmentAngle(self, referenceCoordinate, coordinate):
        """ Method to calculate the relative angle formed by the x-axis and the line defined by one coordinate and the
        reference one. """

        relativeCoordinate = [coordinate[0] - referenceCoordinate[0], coordinate[1] - referenceCoordinate[1]]

        sine = abs(relativeCoordinate[1]) / self.getDistance2D(np.array(referenceCoordinate),
                                                               np.array(coordinate))

        if relativeCoordinate[1] >= 0:

            # First Quadrant
            if relativeCoordinate[0] >= 0:
                angle = math.asin(sine)

                return np.rad2deg(angle)

            # Second Quadrant
            else:
                angle = math.pi - math.asin(sine)

                return np.rad2deg(angle)
        else:

            # Fourth Quadrant
            if relativeCoordinate[0] >= 0:
                angle = (2 * math.pi) - math.asin(sine)

                return np.rad2deg(angle)

            # Third Quadrant
            else:
                angle = math.pi + math.asin(sine)

                return np.rad2deg(angle)

    def getConvexHullBis(self, vectorMisalignment2D):
        """ Method to calculate the convex hull that contains all the misalignment coordinates. Each coordinate is
        described by the module and direction of the misalignment introduced in each tilt-image. In order to do so,
        the algorithm will pick the coordinate with the smallest component as a initial point. Then, will look for the
        next coordinate that in contained in the hull recursively, starting the process again with each coordinate that
        is added to the hull. The condition for a coordinate to be added to the is that no concave angle may be formed
        with this coordinate and ony other contained in the set."""

        hull = []
        misalignmentCoordinates = vectorMisalignment2D.copy()

        """ The recursion algorithm is started at the coordinate with minimum X component (could be minimum y too). """
        startingCoordinate = sorted(misalignmentCoordinates, key=lambda elem: elem[0])[0]

        hull.append(startingCoordinate)
        remainingCoordinates = misalignmentCoordinates.copy()

        while misalignmentCoordinates:
            coordinate = random.choice(misalignmentCoordinates)

            while remainingCoordinates:
                element = remainingCoordinates[0]
                coordinateVector = [hull[-1][0] - coordinate[0], hull[-1][1] - coordinate[1]]

                elementVector = [element[0] - coordinate[0], element[1] - coordinate[1]]

                angle = np.arctan2(elementVector[1],
                                   elementVector[0]) - np.arctan2(coordinateVector[1],
                                                                  coordinateVector[0])

                if angle < 0:
                    angle += 2 * math.pi

                angle = np.rad2deg(angle)

                if angle < 180:
                    remainingCoordinates.remove(element)
                else:
                    coordinate = element  # comprobar para 180 grados
                    remainingCoordinates.remove(element)

            if coordinate == startingCoordinate:
                break

            hull.append(coordinate)

            misalignmentCoordinates.remove(coordinate)
            remainingCoordinates = misalignmentCoordinates.copy()

        if self.generateOutputHullPlot:
            import matplotlib.pyplot as plt
            plt.scatter(*zip(*vectorMisalignment2D))
            plt.scatter(*zip(*hull))
            plt.show()

        return hull

    @staticmethod
    def getHullArea(hull):
        """ Method to calculate the area occupied by the convex hull in which set of coordinates that describes the
        misalignment introduced for each 3D coordinate at each projection through the tilt-series are contained. As
        a convex polygon, its area is calculate applying the shoelace formula. """

        summary1 = 0
        summary2 = 0

        for i in range(len(hull)):
            shiftedIndex1 = (i + 1) % len(hull)
            prod1 = hull[i][0] * hull[shiftedIndex1][1]
            summary1 += prod1

        for i in range(len(hull)):
            shiftedIndex2 = (i + 1) % len(hull)
            prod2 = hull[shiftedIndex2][0] * hull[i][1]
            summary2 += prod2

        area = abs(1 / 2 * (summary1 - summary2))

        return ['%.4f' % area]

    def getDistanceHistogram(self, distanceVector):
        """ Method to generate an histogram representation from each distance of each 3D coordinate and its misaligned
        counterpart through the series. """

        binWidth = self.getFreedmanDiaconisBinWidth(distanceVector)
        numberOfBins = int(math.floor((max(distanceVector) - min(distanceVector) / binWidth) + 1))
        histogram, _ = np.histogram(distanceVector, numberOfBins)

        if self.generateOutputHistogramPlots:
            import matplotlib.pyplot as plt
            plt.hist(distanceVector, numberOfBins)
            plt.show()

        binWidth = self.getSquareRootBinWidth(distanceVector)
        numberOfBins = int(math.floor((max(distanceVector) - min(distanceVector) / binWidth) + 1))
        histogram, _ = np.histogram(distanceVector, numberOfBins)

        if self.generateOutputHistogramPlots:
            import matplotlib.pyplot as plt
            plt.hist(distanceVector, numberOfBins)
            plt.show()

        binWidth = self.getSturguesBinWidth(distanceVector)
        numberOfBins = int(math.floor((max(distanceVector) - min(distanceVector) / binWidth) + 1))
        histogram, _ = np.histogram(distanceVector, numberOfBins)

        if self.generateOutputHistogramPlots:
            import matplotlib.pyplot as plt
            plt.hist(distanceVector, numberOfBins)
            plt.show()

        binWidth = self.getRiceBinWidth(distanceVector)
        numberOfBins = int(math.floor((max(distanceVector) - min(distanceVector) / binWidth) + 1))
        histogram, _ = np.histogram(distanceVector, numberOfBins)

        if self.generateOutputHistogramPlots:
            import matplotlib.pyplot as plt
            plt.hist(distanceVector, numberOfBins)
            plt.show()

        return histogram

    def getDistributionMoments(self, histogram):
        """ Method to calculate the first n moments of distance distribution histogram. """

        moments = []

        for order in range(1, self.maximumOrderMoment + 1):
            moments.append('%.4f' % scipy.stats.moment(histogram, order))

        return moments


# ----------------------------------- Main ------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prepareDataset.py <pathCoordinate3D> <pathAngleFile> <pathMisalignmentMatrix> "
              "<pathPatternToSubtomoFiles>\n"
              "<pathCoordinate3D>: Path to folder containing the 3D coordinate files belonging to the same series in "
              "Xmipp format (xmd). \n"
              "<pathAngleFile>: Path to folder containing the tilt angle files of the tilt-series. \n"
              "<pathMisalignmentMatrix>: Path to the folder containing the misalignment matrix files for each "
              "tilt-image from the series. \n"
              "<pathPatternToSubtomoFiles>: Path to the folder containing the subtomo volume files")
        exit()

    td = TrackerDisplacement(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
