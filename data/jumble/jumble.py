from random import shuffle

lines = open('gold').readlines()
for i, line in enumerate(lines):
    words = line.strip().split(' ')
    jumbles = []
    jumbles.append(' '.join(words))
    for i in xrange(9):
        shuffle(words)
        jumbles.append(' '.join(words))

    f = open('test%s' % i, mode='w')
    shuffle(jumbles)
    for word in jumbles:
        f.write(' '.join(words) + '\n')
    f.close()
    
