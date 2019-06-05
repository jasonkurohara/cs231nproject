import os
import shutil
#########################################CONSTANTS##################################################
IMG_WIDTH = 1024.0
IMG_HEIGHT = 540.0

#########################################FUNCTIONS##################################################

def getImgName(current_frame, filename):
  if current_frame >= 1000:
    return "./annotations/" + filename[:5] + "_img00" + str(current_frame) + ".txt"
  elif current_frame >= 100:
    return "./annotations/" + filename[:5] + "_img000" + str(current_frame) + ".txt"
  elif current_frame >= 10:
     return "./annotations/" + filename[:5] + "_img0000" + str(current_frame) + ".txt"
  else:
    return "./annotations/" + filename[:5] + "_img00000" + str(current_frame) + ".txt"

def getBBoxes(content, filename):

  k = len(content)
  #total_frames = int((content[:].split(',')[0])) #get total number of frames
 # print("MAX: ", total_frames)
  print(filename)
  #print(k)

  prev_frame = 1
  prev_name = ""
  #print(total_frames)
  for i in range(k):
    
   
    current_frame = int(content[i].split(',')[0]) #get current frame
    name = getImgName(current_frame, filename)



    
    if os.path.exists(name):
      txt_file= open(name,"a")
    else:
      txt_file= open(name,"w+")
  
    box_left = content[i].split(',')[2]
    box_top = content[i].split(',')[3]
    width = float(content[i].split(',')[4])
    height = float(content[i].split(',')[5])

    centerx = float(box_left) + (width//2)
    centery = float(box_top) + (height//2)

    txt_file.write("0" + ' ' + str(centerx/IMG_WIDTH) + " " + str(centery/IMG_HEIGHT) + " " + str(width/IMG_WIDTH) + " " + str(height/IMG_HEIGHT) + '\n')
  
#ßdef removeSparse():


####################################################################################################

directory = os.fsencode('./GT/')
#shutil.rmtree("./annotations")
#os.makedirs("./annotations")
for file in sorted(os.listdir(directory)):
     filename = os.fsdecode(file)
   #  print(filename)
     if "gt_whole.txt" in filename: 
        print("opening...",filename)
        with open("./GT/" + filename, "r") as f: #iterate through each scene
          content = f.read().splitlines()
          getBBoxes(content,filename)
      #    k = len(content)
       #   print(k)

         # print(os.path.join(directory, filename))
        continue
     else:
         continue


         