import os
import sys
import xmippLib as xmipp
import shutil


class DeepDefocus:
    @staticmethod
    def importCTF(fnDir, dataFlag):
        fileList = []
        for file in os.listdir(fnDir):
            if file.endswith("_enhanced_psd.xmp"):
                fnRoot = os.path.join(fnDir, file)
                md = xmipp.MetaData(fnRoot.replace("_xmipp_ctf_enhanced_psd.xmp", "_xmipp_ctf.xmd"))
                objId = md.firstObject()
                dU = md.getValue(xmipp.MDL_CTF_DEFOCUSU, objId)
                dV = md.getValue(xmipp.MDL_CTF_DEFOCUSV, objId)
                kV = md.getValue(xmipp.MDL_CTF_VOLTAGE, objId)
                enabled = md.getValue(xmipp.MDL_ENABLED, objId)
                if dataFlag == 1:
                    if enabled == 1:
                        fileList.append((fnRoot, 0.5*(dU+dV), kV))
                else:
                    fileList.append(fnRoot)
        print("Files read from origin")
        return fileList

    @staticmethod
    def downsampleCTF(fileList, stackDir, subset, dataFlag):
        if dataFlag == 1:
            for file in fileList:
                fnRoot, defocus, kV = file
                fnBase = os.path.split(fnRoot)[1]
                destRoot = stackDir + fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset)
                if os.path.isfile(os.path.join(stackDir, "metadata.txt")):
                    metadataPath = open(os.path.join(stackDir, "metadata.txt"), "r+")
                    metadataLines = metadataPath.read().splitlines()
                    lastLine = metadataLines[-1]
                    i = int(lastLine[0:9]) + 1
                else:
                    metadataPath = open(os.path.join(stackDir, "metadata.txt"), "w+")
                    metadataPath.write("  ID         DEFOCUS      kV   SUBSET  FILE\n")
                    i = 0
                shutil.copy(fnRoot, destRoot)
                metadataPath.write("%9.7d%11d%8d%9d  %s\n" % (i, defocus, kV, subset, destRoot))
                i += 1
            print("Files copied to destiny and metadata generated")

        else:
            for fnRoot in fileList:
                metadataPath = open(os.path.join(stackDir, "metadata.txt"), "r+")
                fnBase = os.path.split(fnRoot)[1]
                destRoot = stackDir + fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset)
                metadataLines = metadataPath.read().splitlines()
                metadataLines.pop(0)
                for line in metadataLines:
                    storedFile = line[40:]
                    storedFileBase = os.path.split(storedFile)[1]
                    if storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_1.xmp") \
                            or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_2.xmp") \
                            or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_3.xmp"):
                        shutil.copy(fnRoot, destRoot)
            print("Files copied to destiny")

    @staticmethod
    def prune(stackDir):
        metadataPath = open(os.path.join(stackDir, "metadata.txt"), "w+")
        lines = metadataPath.read().splitlines()
        lines.pop(0)
        nameFiles = []
        for line in lines:
            fileName = line [40:-1]
            nameFiles.append(fileName)

        for file in stackDir:
            for nameFile in nameFiles:
                nameFile.replace()
                if nameFile == file:
                    break
                file.replace(".xmp", "_ERRASE.xmp")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python prepareDataset.py <dirIn> <dirOut> <subsetNumber> <importDataFlag(0/1)>")
    fnDir = sys.argv[1]
    stackDir = sys.argv[2]
    subset = int(sys.argv[3])
    dataFlag = int(sys.argv[4])
    stackDir = stackDir + "/"
    deepDefocus = DeepDefocus()
    allPSDs = deepDefocus.importCTF(fnDir, dataFlag)
    deepDefocus.downsampleCTF(allPSDs, stackDir, subset, dataFlag)
