"""CPU-only multi-process aiohttp load generator; every HTTP response validated."""
import asyncio,time,random
from pathlib import Path
def run_worker(args):
 url,paths,total,conc,seed,start_at=args
 blobs=[Path(p).read_bytes() for p in paths]
 ids=list(range(total));random.Random(seed).shuffle(ids)
 async def go():
  import aiohttp
  while time.perf_counter()<start_at:await asyncio.sleep(.002)
  records=[];counter=iter(ids)
  async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc),timeout=aiohttp.ClientTimeout(total=120)) as session:
   async def work():
    for idx in counter:
     t=time.perf_counter()
     async with session.post(url+'/predict',data=blobs[idx%len(blobs)],headers={'Content-Type':'image/png'}) as r:
      data=await r.json()
      if r.status!=200 or 'detections' not in data:raise RuntimeError(f'HTTP {r.status}: {data}')
     end=time.perf_counter();records.append(dict(start=t,end=end,ms=(end-t)*1000,boxes=len(data['detections'])))
   await asyncio.gather(*(work() for _ in range(conc)))
  return records
 return asyncio.run(go())
