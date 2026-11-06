import sys, numpy as np

import minitrack as mt, brfit as bf
from validate_v14 import v4_sample
r, vtx, p1, p2, h1, h2 = v4_sample()
n=300; h1,h2,vtx,p1,p2=h1[:n],h2[:n],vtx[:n],p1[:n],p2[:n]
o=mt.vertex_fit(h1,h2,vtx,p1,p2,parallel=True,cov_fn=bf.make_cov_b(False),pred_fn=bf.make_pred_b(False),return_state=True)
st=o['state']; c0=st['chi2']; P=st['par']
H=[st['h1'],st['h2']]; W=[st['W1'],st['W2']]
best=np.zeros(len(P))
for dl in (-2e-3,-1e-3,-5e-4,-2e-4,-1e-4,-5e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3):
    p=P.copy(); p[:,4]+=dl
    par,c,_=mt.gauss_newton(p,H,W,free=[0,1,2,3,5],preds=st['preds'],n_iter=20)
    best=np.maximum(best,c0-c)
    print('dphi2 %+.0e  median(c0-c) %.3f  frac improved %.3f'%(dl,np.median(c0-c),np.mean(c0-c>1e-6)),flush=True)
print('profile best dchi2: median %.3f  f>3.84 %.3f'%(np.median(best),np.mean(best>3.84)))
