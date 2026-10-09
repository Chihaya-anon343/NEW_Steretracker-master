"""Archive selected lawful public PDF sources; no contact email is stored."""
from pathlib import Path
import urllib.request
import json
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor

SESSION=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
SOURCES={
    "Fractal2019":"https://burjcdigital.urjc.es/bitstreams/6cc6b966-a4c5-460a-b33f-ef710eb576b6/download",
    "IPPE2014":"https://3dvar.com/Collins2014Infinitesimal.pdf",
    "Wu2012":"https://media.ee.ntu.edu.tw/personal/pcwu/research/icme2012_spe/icme2012_spe.pdf",
    "Wu2014":"https://media.ee.ntu.edu.tw/personal/pcwu/research/mmsp2014_spe/mmsp2014_spe.pdf",
    "ICGplus2023":"https://arxiv.org/pdf/2302.11458",
    "STag2019":"https://arxiv.org/pdf/1707.06292",
    "CALTag2010":"https://diglib.eg.org/bitstreams/8c1bf0fc-2bab-49e8-9457-c1cce5d3196d/download",
    "AKAZE2013":"https://encov.ip.uca.fr/publications/pubfiles/2013_Alcantarilla_etal_BMVC_akaze.pdf",
    "Lynen2013":"https://www.pure.ed.ac.uk/ws/portalfiles/portal/17220481/Lynen_et_al_2013_A_Robust_and_Modular_Mutli_Sensor_Fusion.pdf",
    "Lv2024":"https://mdpi-res.com/d_attachment/sensors/sensors-24-01645/article_deploy/sensors-24-01645.pdf"
}

def archive(item):
    key,url=item
    target=SESSION/"papers"/(key+".pdf")
    result={"key":key,"url":url,"accessed":"2026-09-30"}
    try:
        if target.exists(): data=target.read_bytes()
        else:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=25) as response:data=response.read()
        if b"%PDF-" not in data[:1024]:raise ValueError("Response is not a PDF")
        if not target.exists():target.write_bytes(data)
        result.update(status="archived",path=str(target),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:result.update(status="source_link_only",error=str(exc))
    return result

if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(archive,SOURCES.items()))
    (SESSION/"papers/archive-manifest.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{k:r.get(k) for k in ["key","status","bytes","error"]} for r in results],ensure_ascii=False))
