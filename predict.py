# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
from typing import List

MODEL_CACHE = "models"
MODEL_URL = "https://weights.replicate.delivery/default/zfturbo/mvsep-mdx23.tar"
CHECKPOINTS_CACHE = "./cache/torch/hub/checkpoints/"
CHECKPOINTS_URL = "https://weights.replicate.delivery/default/zfturbo/mvsep-mdx23-checkpoints.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest, exist_ok=True)
    
    # Use requests to download the file
    import requests
    import tarfile
    import tempfile
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
    
    # Extract the tar file
    with tarfile.open(tmp_file_path, 'r:*') as tar:
        tar.extractall(dest)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download models
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(CHECKPOINTS_CACHE):
            download_weights(CHECKPOINTS_URL, CHECKPOINTS_CACHE)

    def predict(
        self,
        audio: Path = Input(description="Input Audio File"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Clear past runs
        output_folder = "/tmp/results/"
        # Remove output folder if it exists
        if os.path.exists(output_folder):
            os.system("rm -rf " + output_folder)
        # Run MVSEP subprocess
        subprocess.run(["python", "inference.py", "--input_audio", str(audio), "--output_folder", output_folder], check=True)
        # Get list of files in the output folder
        files = os.listdir(output_folder)
        # Return list of files
        output_files = [Path(os.path.join(output_folder, file)) for file in files]
        return output_files

# Local testing script
if __name__ == "__main__":
    import sys
    import requests
    from pathlib import Path
    
    # URL to test with
    test_url = "" # ADD YOUR TEST URL HERE
    
    # Download the audio file locally
    print(f"Downloading audio from: {test_url}")
    response = requests.get(test_url)
    response.raise_for_status()
    
    # Save to temporary file
    temp_audio_path = "/tmp/test_audio.m4a"
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)
    
    print(f"Audio saved to: {temp_audio_path}")
    
    # Create predictor instance and run
    predictor = Predictor()
    print("Setting up model...")
    predictor.setup()
    
    print("Running prediction...")
    result = predictor.predict(audio=Path(temp_audio_path))
    
    print(f"Prediction complete! Output files:")
    for file_path in result:
        print(f"  - {file_path}")