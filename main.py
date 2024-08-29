import pynini
from src.FST.transducers import *
import time



def main():

    typ = input('FST : [tex|gram|gram4] : ')
    while(not typ in ['tex', 'gram', 'gram4']) : 
        print('   > Invalid')
        typ =  input('FST : [tex|gram|gram4]')

    print("Construct transducer")
    t = time.time()
    if typ == 'tex' : fst = LexMathTransducer()
    elif typ == 'gram' : fst = LexGraOneLayerFST()
    elif typ == 'gram4' : fst = LexGraMultiLayerFST()
    print(f'    >> Done : {time.time() - t}s')

    i = input("sentence | break : ")
    while (i != 'break'):
        print(fst.predict(i))
        i = input("sentence | break : ")



if __name__ == '__main__':
    main()