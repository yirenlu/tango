import string

L = string.letters + list('<spc>')
g = open('fstoneone.txt', 'w+')
for one_letter in L:
	fstText = '0' + ' '*5 + '1' + ' '*5 + one_letter + ' '*5 + one_letter + '\n'
	g.write(fstText)
g.close()