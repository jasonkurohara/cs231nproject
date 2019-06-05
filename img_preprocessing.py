
IMG_WIDTH = 1024.0
IMG_HEIGHT = 540.0
with open("M0101_gt_whole_prepreprocessed.txt", "r") as f:
    content = f.read().splitlines()
    k = len(content)

f= open("img000001.txt","w+")

for i in range(k):
      #  print(content[i].split(',')[0])
        if content[i].split(',')[0] == '1':
           # frame  = content[i].split(',')[1]
            box_left = content[i].split(',')[2]
            box_top = content[i].split(',')[3]
            width = float(content[i].split(',')[4])
            height = float(content[i].split(',')[5])
            # print(box_top)
            centerx = float(box_left) + (width//2)
            centery = float(box_top) + (height//2)

            f.write("0" + ' ' + str(centerx/IMG_WIDTH) + " " + str(centery/IMG_HEIGHT) + " " + str(width/IMG_WIDTH) + " " + str(height/IMG_HEIGHT) + '\n')
                     





for i in range(9):
  file_name = 201 + i
  file_name = "M0" + str(file_name) + "_gt_whole copy.txt"
  print('opening ', file_name, '...')
  with open(file_name, "r") as f:
    content = f.read().splitlines()
    k = len(content)

  scene_name = 2 + i
  if scene_name >= 10:
    scene_name = str(scene_name) + ".txt"
    f= open("img0000" + scene_name,"w+")

  else:
    scene_name = str(scene_name) + ".txt"
    f= open("img00000" + scene_name,"w+")

  for i in range(k):
      #  print(content[i].split(',')[0])
        if content[i].split(',')[0] == '1':
           # frame  = content[i].split(',')[1]
            box_left = content[i].split(',')[2] #top left, xcoord
            box_top = content[i].split(',')[3] #top left, y coord


            width = float(content[i].split(',')[4]) #width of box
            height = float(content[i].split(',')[5]) #height of box
            # print(box_top)
            centerx = float(box_left) + (width//2)
            centery = float(box_top) + (height//2)

         #   print(width)
          #  print(height)
            if((float(box_left) + width//2) >= IMG_HEIGHT):
              assert "asfasd"
           # print("Top Left Y: ", str(box_left))
          #  print("Center X: ", str(centerx))
          #  print("Center Y: ", str(centery))

            if(centerx/IMG_WIDTH) > 1:
              assert "WRONG"
            if(float(box_top)/IMG_HEIGHT) > 1:
              print("PENIS FACE")
       #     print(centery/IMG_HEIGHT)
         #   print("writing...", scene_name)
            f.write("0" + ' ' + str(centerx/IMG_WIDTH) + " " + str(centery/IMG_HEIGHT) + " " + str(width/IMG_WIDTH) + " " + str(height/IMG_HEIGHT) + '\n')
                     


for i in range(9):
  file_name = 201 + i
  file_name = "M0" + str(file_name) + "_gt_whole copy.txt"
  print('opening ', file_name, '...')
  with open(file_name, "r") as f:
    content = f.read().splitlines()
    k = len(content)

  scene_name = 2 + i
  if scene_name >= 10:
    scene_name = str(scene_name) + ".txt"
    f= open("img0000" + scene_name,"w+")

  else:
    scene_name = str(scene_name) + ".txt"
    f= open("img00000" + scene_name,"w+")

  for i in range(k):
      #  print(content[i].split(',')[0])
        if content[i].split(',')[0] == '1':
           # frame  = content[i].split(',')[1]
            box_left = content[i].split(',')[2] #top left, xcoord
            box_top = content[i].split(',')[3] #top left, y coord


            width = float(content[i].split(',')[4]) #width of box
            height = float(content[i].split(',')[5]) #height of box
            # print(box_top)
            centerx = float(box_left) + (width//2)
            centery = float(box_top) + (height//2)

         #   print(width)
          #  print(height)
            if((float(box_left) + width//2) >= IMG_HEIGHT):
              assert "asfasd"
           # print("Top Left Y: ", str(box_left))
          #  print("Center X: ", str(centerx))
          #  print("Center Y: ", str(centery))

            if(centerx/IMG_WIDTH) > 1:
              assert "WRONG"
            if(float(box_top)/IMG_HEIGHT) > 1:
              print("PENIS FACE")
       #     print(centery/IMG_HEIGHT)
         #   print("writing...", scene_name)
            f.write("0" + ' ' + str(centerx/IMG_WIDTH) + " " + str(centery/IMG_HEIGHT) + " " + str(width/IMG_WIDTH) + " " + str(height/IMG_HEIGHT) + '\n')
             

with open("M1401_gt_whole copy.txt", "r") as f:
    content = f.read().splitlines()
    k = len(content)

f= open("imgval.txt","w+")

for i in range(k):
      #  print(content[i].split(',')[0])
        if content[i].split(',')[0] == '1':
           # frame  = content[i].split(',')[1]
            box_left = content[i].split(',')[2]
            box_top = content[i].split(',')[3]
            width = float(content[i].split(',')[4])
            height = float(content[i].split(',')[5])
            # print(box_top)
            centerx = float(box_left) + (width//2)
            centery = float(box_top) + (height//2)

            f.write("0" + ' ' + str(centerx/IMG_WIDTH) + " " + str(centery/IMG_HEIGHT) + " " + str(width/IMG_WIDTH) + " " + str(height/IMG_HEIGHT) + '\n')
                     





