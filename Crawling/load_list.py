def getList():
    with open("C:/ir_files/linksp.txt", 'r') as file:
        links = [line.strip() for line in file]
        return  links
