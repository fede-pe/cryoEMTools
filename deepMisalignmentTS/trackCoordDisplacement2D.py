import numpy as np
import sys


class TrackerDisplacement:
    def __init__(self, pathCoordinate3D, pathAngles, pathMisalignmentMatrix):
        self.getXYCoordinatesMatrix = [[1, 0, 0],
                                       [0, 1, 0]]
        coordinates3D = self.readCoordinates3D(pathCoordinate3D)
        angles = self.readAngleFile(pathAngles)

        misalignmentMatrices = self.readMisalignmentMatrix(pathMisalignmentMatrix)

        for indexCoord, coordinate3D in enumerate(coordinates3D):
            for indexAngle, angle in enumerate(angles):
                projectedCoordinate2D = self.getProjectedCoordinate2D(angle,
                                                                      coordinate3D)

                misalignedProjectedCoordinate2D = \
                    self.getMisalignedProjectedCoordinate2D(angle,
                                                            coordinate3D,
                                                            misalignmentMatrices[:, :, indexAngle])

                print(projectedCoordinate2D, misalignedProjectedCoordinate2D)

    def getProjectedCoordinate2D(self, angle, coordinate3D):
        rotationMatrix = self.getRotationMatrix(angle)

        projectedCoordinate2D = np.matmul(self.getXYCoordinatesMatrix,
                                          np.matmul(rotationMatrix,
                                                    coordinate3D))

        return projectedCoordinate2D

    def getMisalignedProjectedCoordinate2D(self, angle, coordinate3D, misalignmentMatrix):
        rotationMatrix = self.getRotationMatrix(angle)

        projectedCoordiante2D = np.matmul(self.getXYCoordinatesMatrix,
                                          np.matmul(rotationMatrix,
                                                    coordinate3D))

        misalignedProjectedCoordinate2D = np.matmul(misalignmentMatrix,
                                                    [projectedCoordiante2D[0],
                                                     projectedCoordiante2D[1],
                                                     0])

        return [misalignedProjectedCoordinate2D[0], misalignedProjectedCoordinate2D[1]]

    # ----------------------------------- Utils methods -----------------------------------

    @staticmethod
    def getRotationMatrix(angle):
        """ Method to calculate the 3D rotation matrix of a plane given its tilt angle """
        angleRad = np.deg2rad(angle)

        rotationMatrix = [[np.cos(angleRad), 0, np.sin(angleRad)],
                          [0, 1, 0],
                          [-np.sin(angleRad), 0, np.cos(angleRad)]]

        return rotationMatrix

    @staticmethod
    def getProjectionMatrix(angle):
        """ Method to calculate the projection matrix of a plane given its tilt angle """
        angleRad = np.deg2rad(angle)

        # plane matrix
        v1 = [0, 1, 0]
        v2 = [np.cos(angleRad), 0, np.sin(angleRad)]
        v = np.matrix.transpose(np.array([v1, v2]))

        # projection matrix vProj = v (vt v)^-1 vt
        vt = np.matrix.transpose(v)
        vinv = np.linalg.inv(np.matmul(vt, v))

        vProj = np.matmul(v, np.matmul(vinv, vt))

        return vProj

    def calculateDisplacement2D(self, coordinate2D, transformationMatrix):
        """ Method to calculate the distance between a point and it misaligned correspondent """
        homoCoordinate2D = [coordinate2D[0], coordinate2D[1], 1]
        displacedCoordinate2D = self.getDisplacedCoordinate(homoCoordinate2D, transformationMatrix)
        diffVector = homoCoordinate2D - displacedCoordinate2D
        diffVector = [i ** 2 for i in diffVector]
        distance = sum(diffVector)

        return distance

    @staticmethod
    def getDisplacedCoordinate(coordinate2D, transformationMatrix):
        """ Method to calculate the final position of a 2D coordinate after misalignment """

        return np.dot(transformationMatrix, coordinate2D)

    # ----------------------------------- I/O methods -----------------------------------

    @staticmethod
    def readCoordinates3D(filePath):
        """ Method to read 3D coordinate files in IMOD format """
        coordinates = []
        with open(filePath) as f:
            lines = f.readlines()
            for line in lines:
                vector = line.split()
                coordinate = ([float(vector[1]),
                               float(vector[2]),
                               float(vector[3])])
                coordinates.append(coordinate)

        return coordinates

    @staticmethod
    def readAngleFile(filePath):
        """ Method to read angles in .tlt format """
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
        transformation matrices for each tilt-image belonging to the tilt-series """
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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepareDataset.py <pathCoordinate3D> <pathAngleFile> <pathMisalignmentMatrix>\n"
              "<pathCoordinate3D>: Path to file containing the 3D coordinates belonging to the same series in IMOD "
              "format. \n"
              "<pathAngleFile>: Path to file containing the tilt angles of the tilt-series. \n"
              "<pathMisalignmentMatrix>: Path to file containing the misalignment matrices for each tilt-image from "
              "the series. /n")
        exit()

    td = TrackerDisplacement(sys.argv[1], sys.argv[2], sys.argv[3])
