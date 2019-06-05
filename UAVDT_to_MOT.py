def main():
    f= open("M1101_gt_whole.txt","r")
    f1 = f.readlines()
    g = open("gt.txt", "w+")
    for i in f1:
        x = i.split(",")
        g.write(x[0]+","+x[1]+","+x[2]+","+x[3]+","+x[4]+","+x[5]+",1,-1,-1,-1\n")
    f.close()
    g.close()
    
    #Open the file back and read the contents
    #f=open("guru99.txt", "r")
    #if f.mode == 'r':
    #   contents =f.read()
    #    print (contents)
    #or, readlines reads the individual line into a list
    #fl =f.readlines()
    #for x in fl:
    #print(x)
    
if __name__== "__main__":
  main()
