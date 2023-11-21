import argparse
import re
import glob


class GeneratorPhantomCTF:
    def __init__(self):
        self.input_files = []
        self.number_ctf = None

        # Parameters
        self._ctfSamplingRate = []
        self._ctfVoltage = []
        self._ctfDefocusU = []
        self._ctfSphericalAberration = []
        self._ctfChromaticAberration = []
        self._ctfEnergyLoss = []
        self._ctfLensStability = []
        self._ctfConvergenceCone = []
        self._ctfLongitudinalDisplacement = []
        self._ctfTransversalDisplacement = []
        self._ctfQ0 = []
        self._ctfK = []
        self._ctfEnvR0 = []
        self._ctfEnvR1 = []
        self._ctfEnvR2 = []
        self._ctfBgGaussianK = []
        self._ctfBgGaussianSigmaU = []
        self._ctfBgGaussianCU = []
        self._ctfBgSqrtK = []
        self._ctfBgSqrtU = []
        self._ctfBgBaseline = []
        self._ctfBgGaussian2K = []
        self._ctfBgGaussian2SigmaU = []
        self._ctfBgGaussian2CU = []
        self._ctfBgR1 = []
        self._ctfBgR2 = []
        self._ctfBgR3 = []
        self._ctfDefocusV = []
        self._ctfDefocusAngle = []
        self._ctfBgGaussianSigmaV = []
        self._ctfBgGaussianCV = []
        self._ctfBgGaussianAngle = []
        self._ctfBgSqrtV = []
        self._ctfBgSqrtAngle = []
        self._ctfBgGaussian2SigmaV = []
        self._ctfBgGaussian2CV = []
        self._ctfBgGaussian2Angle = []
        self._ctfX0 = []
        self._ctfXF = []
        self._ctfY0 = []
        self._ctfYF = []
        self._ctfCritFitting = []
        self._ctfCritCorr13 = []
        self._ctfCritIceness = []
        self._CtfDownsampleFactor = []
        self._ctfCritPsdStdQ = []
        self._ctfCritPsdPCA1 = []
        self._ctfCritPsdPCARuns = []

        self.fields = ['_ctfSamplingRate', '_ctfVoltage', '_ctfDefocusU', '_ctfSphericalAberration', '_ctfChromaticAberration', '_ctfEnergyLoss', '_ctfLensStability', '_ctfConvergenceCone', '_ctfLongitudinalDisplacement', '_ctfTransversalDisplacement', '_ctfQ0', '_ctfK', '_ctfEnvR0', '_ctfEnvR1', '_ctfEnvR2', '_ctfBgGaussianK', '_ctfBgGaussianSigmaU', '_ctfBgGaussianCU', '_ctfBgSqrtK', '_ctfBgSqrtU', '_ctfBgBaseline', '_ctfBgGaussian2K', '_ctfBgGaussian2SigmaU', '_ctfBgGaussian2CU', '_ctfBgR1', '_ctfBgR2', '_ctfBgR3', '_ctfDefocusV', '_ctfDefocusAngle', '_ctfBgGaussianSigmaV', '_ctfBgGaussianCV', '_ctfBgGaussianAngle', '_ctfBgSqrtV', '_ctfBgSqrtAngle', '_ctfBgGaussian2SigmaV', '_ctfBgGaussian2CV', '_ctfBgGaussian2Angle', '_ctfX0', '_ctfXF', '_ctfY0', '_ctfYF', '_ctfCritFitting', '_ctfCritCorr13', '_ctfCritIceness', '_CtfDownsampleFactor', '_ctfCritPsdStdQ', '_ctfCritPsdPCA1', '_ctfCritPsdPCARuns']

        self.readInput()
        self.generatePhantoms()

    def readInput(self):
        parser = argparse.ArgumentParser(description="Generate new CTFs based on a characterized population.")

        # Add the inputFiles parameter with type=str
        parser.add_argument(
            '-inputFiles',
            required=True,
            help='Input CTF parameter files to characterize the population from which new CTFs will be generated. '
                 'Accepts a regex.'
        )

        # Add the numberCTF parameter
        parser.add_argument(
            '-numberCTF',
            type=int,
            required=True,
            help='Number of phantoms CTFs to be generated.'
        )

        # Parse the command line arguments
        args = parser.parse_args()

        # Access the values using args.inputFiles and args.numberCTF
        input_files_regex = args.inputFiles
        self.number_ctf = args.numberCTF

        # Use the re module to work with the regex
        try:
            compiled_regex = re.compile(input_files_regex)
        except re.error:
            print("Invalid regular expression for inputFiles.")
            return False

        # Use glob to get a list of files matching the regex
        self.input_files = glob.glob(input_files_regex)

    def processParamsFiles(self):ç
        pass

    def generatePhantoms(self):
        # Your program logic goes here using self.input_files and self.number_ctf
        pass


if __name__ == '__main__':
    generator = GeneratorPhantomCTF()
