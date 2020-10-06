
import numpy as np


class trackerDisplacement2D:
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

    def readCoordinates3D(self, filePath):
        coordinates = []
        with open(filePath) as f:
            lines = f.readlines()
            for line in lines:
                vector = line.split()
                coordinate = (vector[1], vector[2], vector[3])
                coordinates.append(coordinate)

        return coordinates




