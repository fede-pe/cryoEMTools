
import numpy as np
import sys


class TrackerDisplacement:
    def __init__(self, pathCoordinate3D):
        coordinates3D = self.readCoordinates3D(pathCoordinate3D)
        print(coordinates3D)


    @staticmethod
    def getDisplacedCoordinate(coordinate2D, transformationMatrix):

        return np.dot(transformationMatrix, coordinate2D)

    def calculateDisplacement2D(self, coordinate2D, transformationMatrix):
        homoCoordinate2D = [coordinate2D[0], coordinate2D[1], 1]
        displacedCoordinate2D = self.getDisplacedCoordinate(homoCoordinate2D, transformationMatrix)
        diffVector = homoCoordinate2D - displacedCoordinate2D
        diffVector = [i ** 2 for i in diffVector]
        distance = sum(diffVector)

        return distance

    def getProjectionMatrix(self, angle):
        # plane matrix
        V = np.array([0, 1, 0], [np.cos(angle), 0, np.sin(angle)])
        # projection matrix Vp = V (Vt V)^-1 Vt
        Vp = np.matmul(V, np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(V), V)), np.matrix.transpose(V)))

        return Vp

    @staticmethod
    def readCoordinates3D(filePath):
        coordinates = []
        with open(filePath) as f:
            lines = f.readlines()
            for line in lines:
                vector = line.split()
                coordinate = (vector[1], vector[2], vector[3])
                coordinates.append(coordinate)

        return coordinates

    @staticmethod
    def readAngleFile(filePath):
        angles = []
        with open(filePath) as f:
            lines = f.readlines()
            for line in lines:
                vector = line.split()
                angles.append(vector[0])

        return angles


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepareDataset.py <pathCoordinate3D>\n"
              "<pathCoordinate3D>: Path to a file containing the 3D coordinates belonging to the same series.")

    td = TrackerDisplacement(sys.argv[1])





