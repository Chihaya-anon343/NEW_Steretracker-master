"""Consolidate real screening records and build a shared numbered bibliography.

Run from the project root. Does not modify the original method draft.
"""
from pathlib import Path
import json
import re
import sys
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFS = [
    ("Lee2024", "10.2514/6.2024-0338"),
    ("Xing2019", "10.7527/S1000-6893.2018.22601"),
    ("Olson2011", "10.1109/ICRA.2011.5979561"),
    ("Garrido2014", "10.1016/j.patcog.2014.01.005"),
    ("Fractal2019", "10.1109/ACCESS.2019.2951204"),
    ("Recursive2026", "10.3390/jsan15050070"),
    ("Pressigout2007", "10.1177/0278364907080477"),
    ("UcoSLAM2020", "10.1016/j.patcog.2019.107193"),
    ("Collins2014", "10.1007/s11263-014-0725-5"),
    ("Wu2014", "10.1109/MMSP.2014.6958793"),
    ("Lv2024", "10.3390/s24051645"),
    ("Krogius2019", "10.1109/IROS40897.2019.8967787"),
    ("Yuan2018", "10.1177/1550147718800655"),
    ("Wang2016", "10.1109/IROS.2016.7759617"),
    ("STag2019", "10.1016/j.imavis.2019.06.007"),
    ("AKAZE2013", "10.5244/C.27.13"),
    ("HybridPose2020", "10.1109/CVPR42600.2020.00051"),
    ("ICGplus2023", "10.1109/IROS55552.2023.10341961"),
    ("CALTag2010", "10.2312/PE/VMV/VMV10/041-048"),
    ("RUNE2011", "10.1109/CVPR.2011.5995544"),
    ("EPnP2009", "10.1007/s11263-008-0152-6"),
    ("Chng2020", "10.1109/ICRA40945.2020.9196902"),
    ("Wu2012", "10.1109/ICME.2012.176"),
    ("Lynen2013", "10.1109/IROS.2013.6696917"),
    ("Sola2017", "arxiv:1711.02508"),
]

# Publication records override incomplete or conflicting registry fields.
PRIMARY_OVERRIDES = {
    "10.7527/s1000-6893.2018.22601": {"container-title":["航空学报"],"volume":"40","issue":"6","page":"322601","type":"journal-article","source":"https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2018.22601"},
    "10.1007/s11263-014-0725-5": {"container-title":["International Journal of Computer Vision"],"volume":"109","page":"252-286","type":"journal-article","source":"https://3dvar.com/Collins2014Infinitesimal.pdf"},
    "10.2312/pe/vmv/vmv10/041-048": {"container-title":["Vision, Modeling, and Visualization"],"page":"41-48","type":"proceedings-article","source":"https://diglib.eg.org/items/8aaa0fd8-f07d-45b2-bad2-d38dd31ff035"},
    "10.1109/cvpr42600.2020.00051": {"page":"431-440","source":"https://openaccess.thecvf.com/content_CVPR_2020/html/Song_HybridPose_6D_Object_Pose_Estimation_Under_Hybrid_Representations_CVPR_2020_paper.html","note":"CVF正式排版页码；Crossref/IEEE注册页码428-437另保留在原始缓存，不混用。"},
    "10.1177/1550147718800655": {"page":"","source":"https://journals.sagepub.com/doi/10.1177/1550147718800655","note":"Crossref page字段呈截断文章号，省略该字段，保留卷期与DOI。"},
    "10.5244/c.27.13": {"container-title":["Proceedings of the British Machine Vision Conference"],"source":"https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0013/index.html"}
}

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def fetch_metadata(doi):
    try:
        url = "https://api.crossref.org/works/"+urllib.parse.quote(doi, safe="")
        req = urllib.request.Request(url, headers={"User-Agent":"SteretrackerLiteratureReview/1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            msg=json.load(response)["message"]
        return doi.lower(), {"source":url,"checked_on":"2026-09-30","message":msg}
    except Exception as exc:
        return doi.lower(), {"error":str(exc),"checked_on":"2026-09-30"}

def canon(p):
    if p.get("doi"):
        return re.sub(r"^https://doi\.org/", "", p["doi"].strip().lower())
    for k in ["id","key","full_text_url","url","primary_url"]:
        value=str(p.get(k, ""))
        m=re.search(r"(?:arxiv[:/]|arxiv\.org/(?:abs|pdf|html)/)(\d{4}\.\d+)",value,re.I)
        if m:return "arxiv:"+m.group(1)
    return "title:"+re.sub(r"\W+","",p["title"].lower())

def main():
    cache_path=SESSION/"citations/crossref-metadata.json"
    cache=json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    needed=[doi for _,doi in REFS if not doi.startswith("arxiv:") and doi.lower() not in cache]
    if needed:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for key, value in pool.map(fetch_metadata,needed):cache[key]=value
        write_json(cache_path, cache)

    combined={}; graph=[]; queries=[]; aliases={}
    for file in sorted(SESSION.glob("batch_*.json")):
        batch=json.loads(file.read_text(encoding="utf-8-sig"))
        queries.extend({"batch":file.name,"entry":q} for q in batch.get("queries",[]))
        papers=batch.get("papers",batch.get("records",[]))
        for original in papers:
            key=canon(original)
            aliases[original.get("key",key)]=key
            aliases[original.get("id",key)]=key
            p=dict(original)
            p["review_batch"]=file.name
            p["review_date"]="2026-09-30"
            if not p.get("read_level"):
                p["read_level"]="relevant_sections_full_text" if p.get("has_full_text") else p.get("status","abstract_or_metadata_only")
            if key in combined:
                combined[key].setdefault("duplicate_reviews",[]).append(p)
            else:combined[key]=p
            for edge in p.get("references_verified",[]):
                graph.append({"from":key,"to":edge.get("target") or edge.get("target_title"),"batch":file.name,"evidence":edge.get("location"),"original_edge":edge,"depth":1,"type":"backward_reference"})
        graph.extend({**edge,"batch":file.name} for edge in batch.get("citation_edges",batch.get("verified_citation_edges",[])))

    main_batch=json.loads((SESSION/"main-records.json").read_text(encoding="utf-8"))
    queries.extend({"batch":"main-records.json","entry":q} for q in main_batch["queries"])
    for p in main_batch["papers"]:
        key=canon(p)
        if key in combined:
            combined[key].setdefault("additional_verification",[]).append(p)
        else:combined[key]=p
    graph.extend(main_batch["citation_edges"])

    title_ids={p["title"]:key for key,p in combined.items()}
    normalized_graph=[]; seen_edges=set()
    for edge in graph:
        for side in ["from","to"]:
            value=edge.get(side)
            value=aliases.get(value,title_ids.get(value,value))
            if isinstance(value,str) and (value.lower().startswith("10.") or value.lower().startswith("arxiv:")):value=value.lower()
            edge[side]=value
        edge.setdefault("depth",1)
        edge.setdefault("type","backward_reference")
        pair=(edge.get("from"),edge.get("to"))
        if pair not in seen_edges:normalized_graph.append(edge); seen_edges.add(pair)
    graph=normalized_graph

    for number,(bibkey,doi) in enumerate(REFS,1):
        key=doi.lower()
        if key not in combined:raise ValueError("Missing review: "+doi)
        p=combined[key]
        p["manuscript_reference_number"]=number
        p["bibtex_key"]=bibkey
        meta=cache.get(key,{}).get("message")
        if meta:p["crossref_metadata_verified"]=True
        if key in PRIMARY_OVERRIDES:p["primary_metadata_override"]=PRIMARY_OVERRIDES[key]
    write_json(SESSION/"papers-reviewed.json",combined)
    write_json(SESSION/"relevant-papers.json",{k:p for k,p in combined.items() if p.get("score",0)>=7 or p.get("manuscript_reference_number")})
    write_json(SESSION/"search-log.json",queries)
    write_json(SESSION/"citations/citation-graph.json",{"status":"complete_for_this_targeted_review","max_depth":2,"note":"来源批次原始引文证据保留；书目引用方向from→to。","edges":graph})

    bibliography=["# 引言与相关工作共享参考文献\n", "编号在两章间共享；书目信息经原文、出版记录或 DOI 注册记录核对。本文未指定目标期刊，条目采用通用格式；投稿时按该刊要求统一标点与作者截断规则。\n"]
    bib=[]; mapping=[]
    for n,(bibkey,doi) in enumerate(REFS,1):
        p=combined[doi.lower()]
        m=dict(cache.get(doi.lower(),{}).get("message",{}))
        m.update(PRIMARY_OVERRIDES.get(doi.lower(),{}))
        authors=p["authors"]
        title=p["title"]
        venue="; ".join(m.get("container-title",[])) or p.get("venue","")
        year=p["year"]
        volume=m.get("volume",""); issue=m.get("issue",""); pages=m.get("page","") or m.get("article-number","")
        url="https://arxiv.org/abs/1711.02508" if doi.startswith("arxiv:") else "https://doi.org/"+doi
        typ="EB/OL" if doi.startswith("arxiv:") else "C" if m.get("type")=="proceedings-article" else "J"
        location=venue
        if volume:location+=", "+volume+("("+issue+")" if issue else "")
        if pages:location+=": "+pages
        bibliography.append(f"[{n}] {'; '.join(authors)}. {title}[{typ}]. {location}, {year}. [原文/出版记录]({url})。\n")
        entrytype="misc" if doi.startswith("arxiv:") else "inproceedings" if typ=="C" else "article"
        fields={"author":" and ".join(authors),"title":title,"year":str(year),"url":url}
        if not doi.startswith("arxiv:"):fields["doi"]=doi
        else:fields.update({"eprint":"1711.02508","archivePrefix":"arXiv"})
        if venue:fields["booktitle" if entrytype=="inproceedings" else "journal"]=venue
        if volume:fields["volume"]=volume
        if issue:fields["number"]=issue
        if pages:fields["pages"]=pages.replace("-","--")
        bib.append("@"+entrytype+"{"+bibkey+",\n"+",\n".join("  "+k+" = {"+v+"}" for k,v in fields.items())+"\n}\n")
        mapping.append({"number":n,"key":bibkey,"canonical_id":doi.lower(),"title":title,"cited_url":url,"read_level":p.get("read_level"),"has_full_text":p.get("has_full_text",False),"full_text_read_url":(p.get("full_text_url") or p.get("url")) if p.get("has_full_text") else None,"evidence_source_url":p.get("primary_url") or p.get("url"),"metadata_override":PRIMARY_OVERRIDES.get(doi.lower())})
    (ROOT/"manuscript/参考文献_引言与相关工作.md").write_text("\n".join(bibliography),encoding="utf-8")
    (ROOT/"manuscript/references.bib").write_text("\n".join(bib),encoding="utf-8")
    write_json(SESSION/"citations/manuscript-reference-map.json",mapping)
    counts={"unique_candidates":len(combined),"score_at_least_7":sum(p.get("score",0)>=7 for p in combined.values()),"full_text_relevant_sections":sum(p.get("read_level")=="relevant_sections_full_text" for p in combined.values()),"full_text_available":sum(bool(p.get("has_full_text")) for p in combined.values()),"manuscript_references":len(REFS),"citation_edges":len(graph),"crossref_successes":sum("message" in v for v in cache.values()),"queries_recorded":len(queries)}
    write_json(SESSION/"review-counts.json",counts)
    print(json.dumps(counts,ensure_ascii=False))
    print("Crossref failures:", [key for key,value in cache.items() if "message" not in value])

if __name__=="__main__":main()
