"""WP10A A5, POST-HOC diagnostic on the 48 DEVELOPMENT users only (not part of any declared study; written after the development result was read, before the evaluation was analysed). usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a5_diag_posthoc.py"""
import json, sys, numpy as np
sys.path.insert(0,"integration_report/wp10a/scripts")
import a5_analyze as X, a5_monitors as M
recs=[json.loads(l) for l in open("wp10a_evidence/a5_dev_fits.jsonl")]
recs=[r for r in recs if r.get("ok")]
ev_total=0; ev_trans=0; kurt=[]; levels=[]; lag=[]; alarm_users=0
drops=[]; zs=[]
for r in recs:
    sc,post,prev,tr,trp=X.series_of(r,"none")
    res=M.run_candidate("M0",sc,post,tr,trp,prev)
    ev=res["events"]
    alarm_users+=bool(ev)
    k=M.decoded(np.vstack([prev[None,:],post]))
    chg=np.zeros(len(sc),bool); chg[:]= (k[1:]!=k[:-1])
    for e in ev:
        ev_total+=1
        d=e["day"]
        lo=max(0,d-14-28); 
        in_window=chg[max(0,d-14):d].any(); in_ref=chg[lo:max(0,d-14)].any()
        ev_trans+=int(in_window or in_ref)
        drops.append(e["drop_sd"]); zs.append(e["z"])
    v=sc[np.isfinite(sc)]
    if len(v)>20:
        c=v-v.mean(); kurt.append(float((c**4).mean()/(c**2).mean()**2))
    levels.append((np.nanmean(sc[:45]),np.nanmean(sc[45:])))
    t=tr[np.isfinite(tr)]
    lag.append((np.nanmean(t), np.nanmean(sc)))
print("users",len(recs),"alarmed",alarm_users,"events",ev_total,"events with a decoded regime change in window or reference",ev_trans)
print("median excess kurtosis-ish (kurtosis, normal=3):",np.median(kurt))
print("median drop sd of events",np.median(drops) if drops else None,"median z",np.median(zs) if zs else None)
print("mean score first 45 held-out days vs last 45:",np.mean([a for a,b in levels]),np.mean([b for a,b in levels]))
print("mean training in-sample score vs held-out mean:",np.mean([a for a,b in lag]),np.mean([b for a,b in lag]))
