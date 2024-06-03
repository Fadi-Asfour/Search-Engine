def getList():
    with open("linksp.txt", 'r') as file:
        links = [line.strip() for line in file]
        return  links
