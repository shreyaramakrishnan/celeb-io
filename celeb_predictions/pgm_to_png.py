# takes pgm directory and duplicates it into output
# (but in png form!)
# usage: python pgm_to_png.py pgm_dir output_dir
import sys
import os
from PIL import Image
from pathlib import Path

try:
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  for dir in Path(input_dir).iterdir():
    if os.path.isdir(dir):
      os.chdir(dir)
      for item in os.listdir("."):
          item = os.path.abspath(item)
          filename, extension  = os.path.splitext(item)
          if extension == ".pgm":
            print("is pgm")
            print(filename)
            filename = os.path.basename(filename)
            filename = os.path.basename(dir) + "_" + filename
            new_file = "{}.png".format(filename)
            output_file_name = output_dir + new_file
            with Image.open(item) as im:
                print(new_file)
                im.save(new_file)

      os.chdir("..")

except Exception as e:
  print(e)
  print(" usage: python pgm_to_png.py pgm_dir output_dir")



