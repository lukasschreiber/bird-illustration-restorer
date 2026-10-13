"""Install the local protection model, verifying the upstream model checksum."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve


URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
MD5 = "4fab47adc4ff364be1713e97b7e66334"


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    destination = Path(__file__).resolve().parents[1] / "data/models/birefnet-general-lite.onnx"
    if destination.is_file() and checksum(destination) == MD5:
        print("Protection model is already installed and verified.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.part")
    print("Downloading the local BiRefNet protection model...")
    urlretrieve(URL, temporary)
    if checksum(temporary) != MD5:
        raise RuntimeError("Model checksum mismatch; downloaded model was not activated.")
    temporary.replace(destination)
    print(f"Installed verified model: {destination}")


if __name__ == "__main__":
    main()
