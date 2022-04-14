import copy

l1 = [1, 4, 9, 16, 25] #a- list definition

erste_ele = l1[0] # erste Element
vorletzt_ele = l1[-2] # vorletzte Element
zweit_viert_ele = l1[1:4] # zweite bis vierte Elemente
l1_add_ele = l1.append(36)
l2 = l1
l2[0] = 100
l3 = copy.deepcopy(l1) # die Kopie von der Liste l1 in der liste l3

# die beide Element wird geändert, der Grund Warum ist dass der l2 auf den l1 zuweist. d.h. er hat die gleiche Adresse wie der liste l1

for index, elem in enumerate(l1):
    print(f"{index} : {elem}")
