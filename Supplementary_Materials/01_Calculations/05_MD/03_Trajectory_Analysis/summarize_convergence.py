
#!/usr/bin/env python3
from pathlib import Path
import csv, json, math, statistics
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; A=R/"analysis"; M=json.loads((A/"analysis_manifest.json").read_text()); chunks=M["chunks_per_replica"]; n_expected=chunks*500; dt=2.0
files={"rmsd":"polymer_rmsd.dat","rg":"polymer_rg.dat","end_to_end":"end_to_end.dat","head_selected":"pfoa_head_selectedN.dat","head_any":"pfoa_head_anyN.dat","pfoa_polymer":"pfoa_polymer_mindist.dat","polymer_image":"polymer_self_image.dat","pfoa_image":"pfoa_self_image.dat","tail_contacts":"pfoa_tail_polymer_heavy_contacts_4A.dat","tail_waters":"pfoa_tail_waters_within_5A.dat"}
labels={"rmsd":"Polymer RMSD (A)","rg":"Polymer Rg (A)","end_to_end":"End-to-end distance (A)","head_selected":"PFOA head-selected N (A)","head_any":"PFOA head-any N (A)","pfoa_polymer":"PFOA-polymer minimum (A)","tail_contacts":"Unique polymer heavy atoms within 4 A of PFOA tail","tail_waters":"Waters within 5 A of PFOA tail"}
key=["rg","end_to_end","head_any","pfoa_polymer"]
def read_series(p):
    v=[]
    for line in p.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        nums=[]
        for x in line.split():
            try: nums.append(float(x))
            except ValueError: pass
        if len(nums)>=2: v.append(nums[1])
    return np.asarray(v,float)
def read_count_series(p, double_column=False):

    rows=[]
    for line in p.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        fields=line.split()
        if len(fields)<(3 if double_column else 2): raise ValueError(f"Incomplete count row in {p}: {line}")
        frame=int(fields[0]); value=float(fields[1])
        if frame!=len(rows)+1 or not np.isfinite(value) or value<0 or not value.is_integer():
            raise ValueError(f"Invalid frame or count in {p}: {line}")
        if double_column and float(fields[2])!=value:
            raise ValueError(f"Watershell 5 A lower/upper mismatch in {p}: {line}")
        rows.append(value)
    return np.asarray(rows,float)
def iat_ess(x):
    x=np.asarray(x,float); n=len(x); y=x-x.mean(); var=np.dot(y,y)/n
    if n<3 or var==0: return 0.5,float(n),np.array([1.0])
    size=1<<(2*n-1).bit_length(); ac=np.fft.irfft(np.fft.rfft(y,size)*np.conjugate(np.fft.rfft(y,size)),size)[:n]; ac/=np.arange(n,0,-1); ac/=ac[0]
    s=0.0
    for value in ac[1:]:
        if value<=0: break
        s+=float(value)
    tau=0.5+s; return tau,min(float(n),n/(2*tau)),ac
data={}; issues=[]
for r in range(1,4):
    rr=f"replica_{r:02d}"; data[rr]={}
    for name,fn in files.items():
        p=A/rr/fn
        v=(read_count_series(p,name=="tail_waters") if name in ("tail_contacts","tail_waters") else read_series(p)); data[rr][name]=v
        if len(v)!=n_expected: issues.append(f"{rr}/{name}: expected {n_expected}, found {len(v)}")
    for name in ["polymer_image","pfoa_image"]:
        if len(data[rr][name]) and float(data[rr][name].min())<9.0: issues.append(f"{rr}/{name}: periodic-image distance below 9 A")
    for name,limit in [("tail_contacts",374),("tail_waters",5163)]:
        v=data[rr][name]
        if len(v) and (float(v.max())>limit or not np.any(v>0)):
            issues.append(f"{rr}/{name}: implausible counts; inspect topology, tail mask, and CPPTRAJ log")
if issues:
    (A/"stage11_summary.json").write_text(json.dumps({"technical_status":"FAIL","issues":issues},indent=2)+"\n"); print("\n".join(issues)); raise SystemExit(1)

per=[]; blocks=[]; diag=[]; autocorr={}
for rr,d in data.items():
    for name,v in d.items():
        tau,ess,ac=iat_ess(v); autocorr[(rr,name)]=ac
        per.append({"replica":rr,"metric":name,"n":len(v),"mean":float(v.mean()),"sd":float(v.std(ddof=1)),"min":float(v.min()),"max":float(v.max()),"iat_frames":tau,"iat_ps":tau*dt,"ess":ess})
        for b in range(chunks):
            w=v[b*500:(b+1)*500]; blocks.append({"replica":rr,"block_ns":b+1,"metric":name,"mean":float(w.mean()),"sd":float(w.std(ddof=1))})
for name in key:
    means=[]; sds=[]; metric_rows=[]
    for rr,d in data.items():
        v=d[name]; first=v[:500]; last=v[-500:]; sd=float(v.std(ddof=1)); shift=abs(float(first.mean()-last.mean()))/sd if sd else 0.0; tau,ess,_=iat_ess(v)
        metric_rows.append({"replica":rr,"metric":name,"ess":ess,"early_mean":float(first.mean()),"late_mean":float(last.mean()),"standardized_shift":shift,"ess_pass":ess>=20,"shift_pass":shift<=0.5})
        means.append(float(v.mean())); sds.append(sd)
    pooled=math.sqrt(sum(x*x for x in sds)/len(sds)); spread=max(means)-min(means); ratio=spread/pooled if pooled else 0.0
    for row in metric_rows: row.update({"replica_mean_range":spread,"pooled_within_sd":pooled,"replica_spread_ratio":ratio,"replica_spread_pass":ratio<=1.0}); diag.append(row)
passes=all(x["ess_pass"] and x["shift_pass"] and x["replica_spread_pass"] for x in diag)
decision=("ADEQUATE_AT_5NS" if passes else "EXTEND_TO_10NS") if chunks==5 else ("DIAGNOSTICS_PASS_AT_10NS" if passes else "MANUAL_REVIEW_AFTER_10NS")

def save_csv(name,rows):
    with (A/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
save_csv("per_replica_summary.csv",per); save_csv("block_summary.csv",blocks); save_csv("convergence_diagnostics.csv",diag)
tail_frames=[]
for rr,d in data.items():
    for i,(contacts,waters) in enumerate(zip(d["tail_contacts"],d["tail_waters"]),start=1):
        tail_frames.append({"replica":rr,"frame":i,"time_ns":i*dt/1000,"tail_polymer_unique_heavy_atoms_within_4A":int(contacts),"tail_water_molecules_within_5A":int(waters)})
save_csv("tail_contact_hydration_by_frame.csv",tail_frames)
contact=[]
for rr,d in data.items():
    v=d["head_any"]
    for b in range(chunks): contact.append({"replica":rr,"block_ns":b+1,"fraction_head_anyN_below_5A":float(np.mean(v[b*500:(b+1)*500]<5.0))})
save_csv("contact_occupancy_by_block.csv",contact)

colors=["#0072B2","#D55E00","#009E73"]
plot_metrics=["rg","end_to_end","head_any","pfoa_polymer"]
fig,axs=plt.subplots(4,1,figsize=(9,10),sharex=True)
for ax,name in zip(axs,plot_metrics):
    for color,(rr,d) in zip(colors,data.items()): ax.plot(np.arange(len(d[name]))*dt/1000,d[name],lw=.7,alpha=.8,label=rr,color=color)
    ax.set_ylabel(labels[name]); ax.grid(alpha=.2)
axs[-1].set_xlabel("Production time (ns)"); axs[0].legend(ncol=3,frameon=False); fig.tight_layout(); fig.savefig(A/"timeseries.png",dpi=300); fig.savefig(A/"timeseries.pdf"); plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True); window=250
for ax,name in zip(axs.flat,plot_metrics):
    for color,(rr,d) in zip(colors,data.items()):
        v=d[name]; run=np.convolve(v,np.ones(window)/window,mode="valid"); t=(np.arange(len(run))+window-1)*dt/1000; ax.plot(t,run,lw=1,label=rr,color=color)
    ax.set_title(labels[name]); ax.grid(alpha=.2)
for ax in axs[-1]: ax.set_xlabel("Production time (ns)")
axs[0,0].legend(frameon=False); fig.suptitle("500 ps running means"); fig.tight_layout(); fig.savefig(A/"running_means.png",dpi=300); fig.savefig(A/"running_means.pdf"); plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
for ax,name in zip(axs.flat,plot_metrics):
    for color,(rr,d) in zip(colors,data.items()): ax.plot(range(1,chunks+1),[d[name][b*500:(b+1)*500].mean() for b in range(chunks)],marker="o",label=rr,color=color)
    ax.set_title(labels[name]); ax.grid(alpha=.2)
for ax in axs[-1]: ax.set_xlabel("1 ns block")
axs[0,0].legend(frameon=False); fig.tight_layout(); fig.savefig(A/"block_means.png",dpi=300); fig.savefig(A/"block_means.pdf"); plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(8,6.5),sharex=True)
for ax,name in zip(axs,["tail_contacts","tail_waters"]):
    for color,(rr,d) in zip(colors,data.items()): ax.plot(range(1,chunks+1),[d[name][b*500:(b+1)*500].mean() for b in range(chunks)],marker="o",label=rr,color=color)
    ax.set_ylabel(labels[name]); ax.grid(alpha=.2)
axs[-1].set_xlabel("1 ns block"); axs[0].legend(ncol=3,frameon=False); fig.tight_layout(); fig.savefig(A/"tail_contact_hydration_blocks.png",dpi=300); fig.savefig(A/"tail_contact_hydration_blocks.pdf"); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5))
for color,(rr,d) in zip(colors,data.items()): ax.plot(range(1,chunks+1),[np.mean(d["head_any"][b*500:(b+1)*500]<5) for b in range(chunks)],marker="o",label=rr,color=color)
ax.set(xlabel="1 ns block",ylabel="Head-any-N contact fraction (<5 A)",ylim=(-.03,1.03)); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(A/"contact_occupancy.png",dpi=300); fig.savefig(A/"contact_occupancy.pdf"); plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
for ax,name in zip(axs.flat,plot_metrics):
    for color,rr in zip(colors,data): ax.plot(np.arange(min(501,len(autocorr[(rr,name)])))*dt,autocorr[(rr,name)][:501],lw=1,label=rr,color=color)
    ax.axhline(0,color="black",lw=.6); ax.set_title(labels[name]); ax.grid(alpha=.2)
for ax in axs[-1]: ax.set_xlabel("Lag (ps)")
axs[0,0].legend(frameon=False); fig.tight_layout(); fig.savefig(A/"autocorrelation.png",dpi=300); fig.savefig(A/"autocorrelation.pdf"); plt.close(fig)

summary={"technical_status":"PASS","production_ns_per_replica":chunks,"replicas":3,"frame_spacing_ps":dt,"convergence_decision":decision,"criteria":{"minimum_ess":20,"maximum_standardized_early_late_shift":0.5,"maximum_replica_spread_ratio":1.0},"diagnostics":diag,"additional_descriptive_metrics":["tail_contacts","tail_waters"],"scientific_interpretation":"prepared associated-state persistence/rearrangement only; no spontaneous binding, chloride displacement, or equilibrium free-energy claim","issues":[]}
(A/"stage11_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
lines=["Stage 11 cross-replica convergence summary",f"technical status: PASS",f"production analyzed: 3 x {chunks} ns",f"convergence decision: {decision}","", "Key diagnostics:"]
for name in key:
    rows=[x for x in diag if x["metric"]==name]; lines.append(f"- {name}: min ESS {min(x['ess'] for x in rows):.1f}; max early-late shift {max(x['standardized_shift'] for x in rows):.3f} SD; replica spread {rows[0]['replica_spread_ratio']:.3f} pooled SD")
lines += ["", "Interpretation: prepared associated-state persistence/rearrangement only.","These diagnostics do not establish spontaneous binding, chloride displacement, or equilibrium binding free energy."]
lines += ["", "Tail contacts and hydration (per-replica means; descriptive, not equilibrium estimates):"]
for name in ["tail_contacts","tail_waters"]:
    rows=[x for x in per if x["metric"]==name]
    lines.append(f"- {name}: "+", ".join(f"replica {i}: {row['mean']:.2f}" for i,row in enumerate(rows,1)))
(A/"stage11_summary.txt").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
