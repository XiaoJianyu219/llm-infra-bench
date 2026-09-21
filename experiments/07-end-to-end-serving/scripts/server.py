"""Single owner execution context; bounded FIFO; one inference thread; raw image HTTP body."""
import asyncio, os, time, collections, contextlib
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from common import stats
class Service:
 def __init__(self):
  self.max_batch=int(os.getenv('MAX_BATCH','1'));self.wait=float(os.getenv('MAX_WAIT_MS','0'))/1000
  self.echo=os.getenv('ECHO','0')=='1';self.queue=asyncio.Queue(maxsize=256)
  self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='inference-owner')
  self.samples=collections.defaultdict(lambda:collections.deque(maxlen=100000));self.count=0;self.errors=0;self.batches=collections.Counter();self.engine=None;self.ready=False
 def setup(self):
  if self.echo:return
  import torch,cv2
  from pipeline import TRTModel
  torch.set_num_threads(4);cv2.setNumThreads(1)
  self.engine=TRTModel(os.environ['ENGINE'])
  for _ in range(24):self.engine(torch.zeros(self.max_batch,3,1024,1024,device='cuda'))
 def run_batch(self,items):
  import torch,numpy as np
  from pipeline import decode,preprocess,nms,restore
  arrays=[];metas=[];timings=[]
  for blob,_,arrival in items:
   t=time.perf_counter();im=decode(blob);d=time.perf_counter();a,meta=preprocess(im);p=time.perf_counter()
   arrays.append(a);metas.append(meta);timings.append(dict(queue_ms=(t-arrival)*1000,decode_ms=(d-t)*1000,preprocess_ms=(p-d)*1000))
  t=time.perf_counter();x=torch.from_numpy(np.stack(arrays)).cuda();torch.cuda.synchronize();h=time.perf_counter()
  y=self.engine(x);torch.cuda.synchronize();f=time.perf_counter()
  dets=[nms(z) for z in y];torch.cuda.synchronize();n=time.perf_counter()
  dets=[z.cpu() for z in dets];torch.cuda.synchronize();d=time.perf_counter()
  out=[]
  for det,meta,timing in zip(dets,metas,timings):
   a=time.perf_counter();v=restore(det,meta).tolist();b=time.perf_counter()
   timing.update(h2d_batch_ms=(h-t)*1000,inference_batch_ms=(f-h)*1000,nms_batch_ms=(n-f)*1000,d2h_batch_ms=(d-n)*1000,restore_ms=(b-a)*1000)
   out.append((dict(detections=v),timing))
  return out
 async def worker(self):
  loop=asyncio.get_running_loop()
  while True:
   first=await self.queue.get()
   if first is None:break
   items=[first];deadline=first[2]+self.wait
   while len(items)<self.max_batch:
    try:
     item=self.queue.get_nowait()
    except asyncio.QueueEmpty:
     left=deadline-time.perf_counter()
     if left<=0:break
     try:item=await asyncio.wait_for(self.queue.get(),left)
     except asyncio.TimeoutError:break
    if item is None:raise RuntimeError('Unexpected shutdown sentinel')
    items.append(item)
   try:
    out=await loop.run_in_executor(self.pool,self.run_batch,items);self.batches[len(items)]+=1
    for (_,future,arrival),(body,timing) in zip(items,out):
     timing['total_ms']=(time.perf_counter()-arrival)*1000
     for k,v in timing.items():self.samples[k].append(v)
     self.count+=1
     if not future.cancelled():future.set_result(body)
   except Exception as e:
    self.errors+=len(items)
    for _,future,_ in items:
     if not future.done():future.set_exception(e)
   finally:
    for _ in items:self.queue.task_done()
 async def start(self):
  await asyncio.get_running_loop().run_in_executor(self.pool,self.setup)
  self.task=asyncio.create_task(self.worker());self.ready=True
 async def stop(self):
  self.ready=False
  await self.queue.join();await self.queue.put(None);await self.task
  def cleanup():
   self.engine=None
   if not self.echo:
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
  await asyncio.get_running_loop().run_in_executor(self.pool,cleanup)
  self.pool.shutdown(wait=True)
service=Service()
@asynccontextmanager
async def lifespan(app):
 await service.start()
 try:yield
 finally:await service.stop()
app=FastAPI(lifespan=lifespan)
@app.get('/healthz')
async def health():return dict(ready=service.ready,max_batch=service.max_batch,max_wait_ms=service.wait*1000)
@app.get('/metrics')
async def metrics():
 return dict(requests=service.count,errors=service.errors,queued=service.queue.qsize(),batch_sizes=dict(service.batches),sample_limit=100000,segments={k:stats(list(v)) for k,v in service.samples.items() if v})
@app.post('/predict')
async def predict(request:Request):
 if not service.ready:raise HTTPException(503,'Not ready')
 body=await request.body()
 if not body or len(body)>16*1024*1024:raise HTTPException(413,'Expected image body between 1 byte and 16 MiB')
 if service.echo:return dict(detections=[[10.,20.,30.,40.,.8,1] for _ in range(100)])
 future=asyncio.get_running_loop().create_future();item=(body,future,time.perf_counter())
 try:service.queue.put_nowait(item)
 except asyncio.QueueFull:raise HTTPException(503,'Queue full')
 try:return await future
 except ValueError as e:raise HTTPException(400,str(e))
