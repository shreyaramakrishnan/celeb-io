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
          # if os.path.isfile(item):
          filename, extension  = os.path.splitext(item)
          # print(filename)
          if extension == ".pgm":
            print("is pgm")
            # dir_name = os.path.dirname(item)
            # print(dir_name)
            # filename = dir_name + "_" + filename
            print(filename)
            filename = os.path.basename(filename)
            # filename = filename[1:]
            # filename = filename.replace(".", "")
            filename = os.path.basename(dir) + "_" + filename
            # print(filename)
            # filename = os.path.abspath(item)
            new_file = "{}.png".format(filename)
            output_file_name = output_dir + new_file
            with Image.open(item) as im:
                print(new_file)
                # im.save(output_dir, new_file)
                im.save(new_file)

      os.chdir("..")


  # for path in Path(input_dir).iterdir():
  #   if path.is_dir():
  #     for item in os.listdir(path):
  #       item = os.path.abspath(item)
  #       # if os.path.isfile(item):
  #       filename, extension  = os.path.splitext(item)
  #       print(extension)
  #       if extension == ".pgm":
  #         print("is pgm")
  #         dir_name = os.path.dirname(item)
  #         print(dir_name)
  #         filename = dir_name + "_" + filename
  #         new_file = "{}.png".format(filename)
  #         with Image.open(item) as im:
  #             print(new_file)
  #             im.save(output_dir, new_file)

  # for subject_dir in os.listdir(input_dir):
  #   d = os.path.join(input_dir, subject_dir)
  #   if (os.path.isdir(d)):
  #     for item in os.listdir(d):
  #       # print(item)
  #       item = os.path.abspath(item)
  #       # if os.path.isfile(item):
  #       filename, extension  = os.path.splitext(item)
  #       print(extension)
  #       if extension == ".pgm":
  #         print("is pgm")
  #         dir_name = os.path.dirname(item)
  #         print(dir_name)
  #         filename = dir_name + "_" + filename
  #         new_file = "{}.png".format(filename)
  #         with Image.open(item) as im:
  #             print(new_file)
  #             im.save(output_dir, new_file)

  # for item in os.walk(input_dir):
  #   # for item in os.listdir(big_dir):
  #     print(item)
  #     # item = item.
  #     if os.path.isfile(item):
  #       filename, extension  = os.path.splitext(item)
  #       if extension == ".pgm":
  #         dir_name = os.path.dirname(item)
  #         print(dir_name)
  #         filename = dir_name + "_" + filename
  #         new_file = "{}.png".format(filename)
  #         with Image.open(item) as im:
  #             print(new_file)
              # im.save(new_file)


except Exception as e:
  print(e)
  print(" usage: python pgm_to_png.py pgm_dir output_dir")



