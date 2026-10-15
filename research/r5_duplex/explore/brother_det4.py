from detlib2 import *
f = lambda d: 'z%.1f pk%.2f gap%.2f (%d,%d)%s' % (d['z'], d['peak'], d['peak']-d['second'], d['dx'], d['dy'], 'B' if d['border'] else '')
for i in range(1, 22):
    A, B = load_r('b%03d' % i), load_r('b%03d' % (i+1))
    v = variants2(A, B)
    print(i, i+1, 'SHEET' if i % 2 else 'cross')
    for k, (a, b) in v.items(): print('   ', k, 'a:', f(a), '| b:', f(b))
