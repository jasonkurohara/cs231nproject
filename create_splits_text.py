import os 

directory = os.fsencode('./scenes/')
#shutil.rmtree("./annotations")
#os.makedirs("./annotations")



#######################TRAINIING SPLIT 40 SCENES############################
new_file = open("train.txt", "w+")
path = "data/custom/images/"
counter = 0
for file in sorted(os.listdir(directory))[:40]:
     filename = os.fsdecode(file)
 #    print(filename)
     images = os.fsencode("./scenes/" +filename)
     for image in sorted(os.listdir(images)):
     	name = os.fsdecode(image)
     	print(name)
     	new_line = path + name + '\n'
     	new_file.write(new_line)
     #	print(new_line)
   		#new_file.write(new_line)
     
     	counter +=1 
    # print(counter)
    # new_line = path + filename + '\n'
   #  new_file.write(new_line)
     
#######################DEV SPLIT 5 SCENES############################
new_file = open("valid.txt", "w+")
path = "data/custom/dev_images/"
counter = 0 
for file in sorted(os.listdir(directory))[40:45]:
     filename = os.fsdecode(file)
  #   print(filename)
     images = os.fsencode("./scenes/" +filename)
     for image in sorted(os.listdir(images)):
     	name = os.fsdecode(image)
   #  	print(name)
     	new_line = path + name + '\n'
     	new_file.write(new_line)
     #	print(new_line)
   		#new_file.write(new_line)
     
     	counter +=1 
#######################TEST SPLIT 5 SCENES############################
new_file = open("test.txt", "w+")
path = "data/custom/test_images/"
counter = 0
for file in sorted(os.listdir(directory))[45:]:
     filename = os.fsdecode(file)
  #   print(filename)
     images = os.fsencode("./scenes/" +filename)
     for image in sorted(os.listdir(images)):
     	name = os.fsdecode(image)
 #    	print(name)
     	new_line = path + name + '\n'
     	new_file.write(new_line)
     #	print(new_line)
   		#new_file.write(new_line)
     
     	counter +=1 