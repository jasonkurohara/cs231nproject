import os 

directory = os.fsencode('./scenes/')
#shutil.rmtree("./annotations")
#os.makedirs("./annotations")
print(len(os.listdir(directory)))
for file in sorted(os.listdir(directory)):
      
     filename = os.fsdecode(file)
     print(filename)
   #  print(type(filename))
  
     path = "./scenes/" + filename + "/"
    # print(path)
     scene = os.fsencode(path)
     for image in sorted(os.listdir(scene)):

      image_name = os.fsdecode(image)
      fuller_path = path + image_name
    #  print(fuller_path)
      new_path = path + filename + "_" + image_name
      print(new_path)
      os.rename(fuller_path, new_path)

from distutils.dir_util import copy_tree
for file in sorted(os.listdir(directory)):
      
     filename = os.fsdecode(file)
   #  print(filename)
   #  print(type(filename))
  
     fromdirectory = "./scenes/" + filename + "/"
     todirectory = "./images"
     copy_tree(fromdirectory, todirectory)

