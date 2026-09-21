import json,os,subprocess
from pathlib import Path
D=Path('/root/autodl-tmp/work/llm-infra-bench/days/day07');R=D/'results'
def main():
 data={'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_count':os.cpu_count(),'uname':os.uname().release,'lscpu':subprocess.check_output(['lscpu'],text=True),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True)}
 for name in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us']:
  p=Path(name)
  if p.exists():data[name]=p.read_text()
 (R/'host_environment.json').write_text(json.dumps(data,indent=2))
 print(json.dumps({k:v for k,v in data.items() if k!='lscpu'}))
if __name__=='__main__':main()
