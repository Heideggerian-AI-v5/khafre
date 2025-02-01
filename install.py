import platform
import subprocess
import sys

opsysname = platform.system()

versionMajor, versionMinor = sys.version_info.major, sys.version_info.minor

if (3 != versionMajor) or (12 < versionMinor) or (9 > versionMinor):
    print("Khafre will only work on python versions 3.9 to 3.12. Newer versions will be supported once pytorch is updated for them. Since this python is %d.%d, Khafre cannot be installed." % (versionMajor, versionMinor))
    sys.exit(0)

if "Windows" == opsysname:
    argsTorch = "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124".split(" ")
elif "Linux" == opsysname:
    argsTorch = "pip3 install torch torchvision torchaudio".split(" ")
elif opsysname in ["Darwin", "MacOS", "iOS", "iPadOS"]:
    print("Python multiprocessing queues are not currently supported on MacOS systems. We are working around this but for now Khafre cannot run on such machines.")
    print("Cannot install Khafre.")
    sys.exit(0)
else:
    print("Unrecognized OS: %s" % opsysname)
    print("Cannot install Khafre.")
    sys.exit(0)

subprocess.run(argsTorch)
subprocess.run("pip3 install transformers==4.40.1".split(" "))
subprocess.run("pip3 install -e .")
print("SUCCESS: khafre should be installed in the current python environment.")
print("Installing dependencies for the scripts in examples. Khafre itself is still usable if these installs fail however.")
subprocess.run("pip3 install -r ./examples/requirements.txt")
