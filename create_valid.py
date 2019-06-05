def main():
    f= open("tM0902.txt","r")
    f1 = f.readlines()
    g = open("valid.txt", "w+")
    for i in f1:
        x = i.split("/")
        y = x[3].split("_")
        g.write(x[0]+'/'+x[1]+'/'+x[2]+'/'+y[1])
    """
    current = 'M0101'
    for i in f1:
        
        
        if y[0] != current:
            current = y[0]
            g.close()
            g = open('t'+y[0]+'.txt', "w+")
        g.write(i)
    """
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
