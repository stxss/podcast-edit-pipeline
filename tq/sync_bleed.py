# Cross-correlation of two mics' levels (rough sync / bleed check) and the first loud transients (the clap).
# Usage (in the work dir): python <repo>/tq/sync_bleed.py <speaker_a> <speaker_b>
import sys, numpy as np
from vadlib import load_db
A_, B_ = sys.argv[1], sys.argv[2]
L,T=load_db(A_),load_db(B_)
N=min(len(L),len(T))
a=(L[:N]-L[:N].mean())/L[:N].std(); b=(T[:N]-T[:N].mean())/T[:N].std()
def corr(lo,hi,rng=300):
    cs=[(off, float(np.mean(a[lo:hi]*b[lo+off:hi+off]))) for off in range(-rng,rng+1)]
    cs.sort(key=lambda x:-x[1]); return cs[:3]
print("global", corr(2000,N-2000))
for t in (120,600,1200,1800,2300):
    if (t+120)*100<N: print(t, corr(t*100, t*100+12000))
# first loud transients (clap)
for nm,d in ((A_,L),(B_,T)):
    idx=np.where(d[:6000]>np.percentile(d,99.5))[0][:5]
    print(nm,"loud early (s):",idx/100, d[idx])
