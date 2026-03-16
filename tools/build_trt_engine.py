import subprocess
import sys


def build_engine(onnx_path, engine_path):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        "--fp16",
        "--workspace=4096",
        f"--saveEngine={engine_path}",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_trt_engine.py model.onnx model.trt")
        sys.exit(1)

    build_engine(sys.argv[1], sys.argv[2])
