from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def main():
 t=json.loads((R/'timing.json').read_text())['summary']
 s=json.loads((R/'sweep_summary.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 tags=['pytorch_cpu','fp16_cpu','fp16_gpu','int8_cpu','int8_gpu','plugin_plugin']
 labels=['PyTorch + CPU NMS','FP16 + CPU NMS','FP16 + GPU NMS','INT8 + CPU NMS','INT8 + GPU NMS','FP16 fused plugin']
 med=[t[k]['e2e']['median'] for k in tags];p99=[t[k]['e2e']['p99'] for k in tags]
 axes[0].barh(labels,med,color='#427aa1',label='Median')
 axes[0].scatter(p99,labels,marker='|',s=160,color='#a53a3a',label='p99')
 axes[0].invert_yaxis();axes[0].set_xlabel('End-to-end latency (ms), no HTTP');axes[0].set_title('Paired timing: same process, 160 repeats')
 axes[0].legend();axes[0].grid(axis='x',alpha=.2)
 colors={1:'#264653',4:'#2a9d8f',8:'#e9c46a',16:'#e76f51'};markers={1:'o',4:'s',16:'^',64:'D'}
 for row in s:
  axes[1].scatter(row['qps'],row['p99'],color=colors[row['batch']],marker=markers[row['concurrency']],s=48,edgecolor='#333333',linewidth=.4)
 axes[1].set_yscale('log');axes[1].set_xlabel('HTTP throughput (QPS)');axes[1].set_ylabel('HTTP p99 latency (ms, log scale)')
 axes[1].set_title('64 configurations; median of 3 runs')
 axes[1].grid(alpha=.2)
 handles=[Line2D([],[],marker='o',ls='',color=v,label=f'Batch max {k}') for k,v in colors.items()]+[Line2D([],[],marker=v,ls='',color='gray',label=f'Concurrency {k}') for k,v in markers.items()]
 axes[1].legend(handles=handles,fontsize=7,ncol=2)
 fig.savefig(R/'day07_overview.png',dpi=180);fig.savefig(R/'day07_overview.pdf');plt.close(fig)
 print('PLOT DONE')
if __name__=='__main__':guarded(main)
