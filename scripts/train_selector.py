"""Train/evaluate the lightweight temporal selector on selector JSONL."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from video_annotator.learned_selector import TemporalTrackSelector, load_selector_rows

def tensorize(row, track_mean, track_std, token_mean, token_std, token_dim):
    tracks = (np.asarray(row["features"], np.float32) - track_mean) / track_std
    max_len = max(1, max((len(x) for x in row["token_sequences"]), default=1))
    dim = token_dim
    padded = np.zeros((len(tracks), max_len, dim), np.float32)
    for i, seq in enumerate(row["token_sequences"]):
        for j, token in enumerate(seq[:max_len]):
            arr=np.asarray(token,np.float32)[:dim]; padded[i,j,:len(arr)] = (arr - token_mean[:len(arr)]) / token_std[:len(arr)]
    text = np.asarray(row["text_embedding"], np.float32)
    return torch.from_numpy(tracks), torch.from_numpy(padded), torch.from_numpy(text), torch.tensor(row["labels"], dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--device",default="cpu"); ap.add_argument("--seed",type=int,default=17); args=ap.parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows=load_selector_rows(args.dataset); train=[r for r in rows if r.get("split")=="train"]; val=[r for r in rows if r.get("split")=="validation"]
    all_tracks=np.concatenate([np.asarray(r["features"],np.float32) for r in train]); raw_tokens=[np.asarray(token,np.float32) for r in train for seq in r["token_sequences"] for token in seq]; token_dim=max(map(len,raw_tokens)); all_tokens=np.stack([np.pad(token,(0,token_dim-len(token))) for token in raw_tokens],axis=0)
    tmean,tstd=all_tracks.mean(0),all_tracks.std(0)+1e-6; kmean,kstd=all_tokens.mean(0),all_tokens.std(0)+1e-6
    model=TemporalTrackSelector(all_tracks.shape[1],token_dim,len(train[0]["text_embedding"])).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    pos=sum(sum(r["labels"]) for r in train); neg=sum(len(r["labels"])-sum(r["labels"]) for r in train); loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.,neg/max(1,pos))],device=args.device))
    for _ in range(args.epochs):
        model.train()
        for r in train:
            x,k,txt,y=tensorize(r,tmean,tstd,kmean,kstd,token_dim); x,k,txt,y=[z.to(args.device) for z in (x,k,txt,y)]; opt.zero_grad(); logits=model(x,k,txt); loss=loss_fn(logits,y)
            pos_logits=logits[y>0.5]; neg_logits=logits[y<0.5]
            if len(pos_logits) and len(neg_logits): loss=loss+0.25*torch.nn.functional.softplus(-(pos_logits[:,None]-neg_logits[None,:])).mean()
            loss.backward(); opt.step()
    model.eval(); results=[]; train_scores=[]
    with torch.no_grad():
        for r in train:
            x,k,txt,y=tensorize(r,tmean,tstd,kmean,kstd,token_dim); train_scores.extend(zip(torch.sigmoid(model(x.to(args.device),k.to(args.device),txt.to(args.device))).cpu().numpy().tolist(),r["labels"]))
        thresholds=np.linspace(.1,.9,17); threshold=max(thresholds,key=lambda q: (sum(int(s>=q and y) for s,y in train_scores)*2)/max(1,sum(int(s>=q) for s,y in train_scores)+sum(int(y) for s,y in train_scores)))
        for r in val:
            x,k,txt,y=tensorize(r,tmean,tstd,kmean,kstd,token_dim); scores=torch.sigmoid(model(x.to(args.device),k.to(args.device),txt.to(args.device))).cpu().numpy(); pred=(scores>=threshold).astype(int); results.append({"video_id":r["video_id"],"expression_id":r["expression_id"],"labels":r["labels"],"scores":scores.tolist(),"predictions":pred.tolist()})
    tp=fp=fn=0
    for r in results:
        for y,p in zip(r["labels"],r["predictions"]): tp+=y and p; fp+=(not y) and p; fn+=y and (not p)
    sweep=[]
    for q in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
        stp=sfp=sfn=0
        for row in results:
            for y,s in zip(row["labels"],row["scores"]): stp+=int(y and s>=q); sfp+=int((not y) and s>=q); sfn+=int(y and s<q)
        sweep.append({"threshold":q,"precision":stp/max(1,stp+sfp),"recall":stp/max(1,stp+sfn)})
    report={"train_samples":len(train),"validation_samples":len(val),"epochs":args.epochs,"seed":args.seed,"device":args.device,"threshold":float(threshold),"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"threshold_sweep":sweep,"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"track_mean":tmean,"track_std":tstd,"token_mean":kmean,"token_std":kstd},args.output.with_suffix(".pt")); args.output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
