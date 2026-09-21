from common import *
from trt_utils import build_engine
def main():
    tag=sys.argv[1]
    cfg=json.loads((R/f'{tag}_config.json').read_text())
    cfg['build']=build_engine(SOD/f'{tag}.onnx',SOD/f'{tag}.engine')
    save(f'{tag}_config.json',cfg)
    print('INT8_BUILD',json.dumps(cfg),flush=True)
if __name__=='__main__': guarded(main)
