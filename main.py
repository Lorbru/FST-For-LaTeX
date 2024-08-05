import pynini
from src.FST.transducers import *
import time



def test():
    

    print("Loading transducer :")
    t = time.time()
    fst = FullTransducer()
    print(f'    >> Done : {time.time() - t}s')

    i = input("sentence | break : ")
    while (i != 'break'):
        print(fst.outputs(i))
        i = input("sentence | break : ")



test()