import re, numpy as np
def load(k):
    rows=[]
    for ln in open(f"scratchpad/lowlevel_curves/{k}_raw.txt"):
        m=re.search(r"frames=(\d+).*?r_step=([\d.]+).*?ep_len=([\d.]+).*?r_ep=([\d.\-]+)",ln)
        if m: rows.append((int(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))))
    return np.array(rows)
def smooth(y,w=15): 
    import numpy as np
    if len(y)<w: return y
    return np.convolve(y,np.ones(w)/w,mode='same')
steps={}
for k in ['fbchunk','eechunk','latent_det']:
    steps[k]=[int(x) for x in open(f"scratchpad/lowlevel_curves/{k}_steps.txt").read().split()]
runs={'FB':'fbchunk','EE':'eechunk','LT':'latent_det'}
data={k:load(k) for k in runs}
# latent converged ep_len = mean of last 10% frames
lt=data['LT']; lt_conv=lt[lt[:,0]>0.9*lt[-1,0],2].mean()
print(f"Latent converged ep_len (last 10%): {lt_conv:.1f}  (final frame {lt[-1,0]/1e9:.2f}B)")
print(f"{'run':>4} {'final_frame':>11} {'final_eplen':>11} {'final_rstep':>11}")
for k,f in runs.items():
    d=data[k]; conv=d[d[:,0]>0.9*d[-1,0],2].mean()
    print(f"{k:>4} {d[-1,0]/1e9:>10.2f}B {conv:>11.1f} {d[d[:,0]>0.9*d[-1,0],1].mean():>11.4f}")
print()
# For FB and EE: find nearest checkpoint step where smoothed ep_len ~= lt_conv
for k in ['FB','EE']:
    d=data[k]; frames=d[:,0]; ep=smooth(d[:,2])
    # earliest frame where smoothed ep reaches lt_conv (rising)
    idx=np.where(ep>=lt_conv)[0]
    if len(idx):
        f_at=frames[idx[0]]
        cand=min(steps[runs[k]], key=lambda s:abs(s-f_at))
        # ep_len at that checkpoint frame
        j=np.argmin(np.abs(frames-cand)); 
        print(f"{k}: reaches latent level ({lt_conv:.0f}) at ~{f_at/1e9:.2f}B -> nearest ckpt {cand} ({cand/1e9:.2f}B), smoothed ep_len there ~{ep[j]:.0f}")
    else:
        print(f"{k}: never reaches {lt_conv:.0f}")
