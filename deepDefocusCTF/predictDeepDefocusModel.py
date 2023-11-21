from tensorflow.keras.models import load_model
import sys
import os



# ------------------------ MAIN PROGRAM -----------------------------
if __name__ == "__main__":
    # ----------- PARSING DATA -------------------
    if len(sys.argv) < 3:
        print("Usage: python3 predictDeepDefocusModel.py <metadataDir> <modelDir>")
        sys.exit()


    metadataDir = sys.argv[1]
    modelDir = sys.argv[2]

    loadModelDir = os.path.join(modelDir, 'model.h5')
    model = load_model(loadModelDir)