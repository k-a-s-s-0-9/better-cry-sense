import numpy as np
from pathlib import Path

# EXACT path based on your layout (notice the 'augmented' folder!)
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CATEGORY = "belly_pain"
MEL_DIR = BASE_DIR / CATEGORY / "mel" 

try:
    # Grab the first file in the mel directory
    sample_file = next(MEL_DIR.glob("*.npy"))
    sample_data = np.load(sample_file)
    
    print(f"📄 Found File: {sample_file.name}")
    print(f"✅ Exact Shape: {sample_data.shape}")
    
except StopIteration:
    print(f"❌ Error: No .npy files found in {MEL_DIR}. Double check the spelling!")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")