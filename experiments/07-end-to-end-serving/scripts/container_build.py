import sys,json
from trt_utils import build_engine
if __name__=='__main__':
 print(json.dumps(build_engine(sys.argv[1],sys.argv[2],dict(min=[1,3,1024,1024],opt=[4,3,1024,1024],max=[16,3,1024,1024]))))
