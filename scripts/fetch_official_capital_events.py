#!/usr/bin/env python3
"""抓取证监会公开的中央汇金/国家队公告事件；事件不是每日交易流水。"""
from __future__ import annotations
import json, re, os
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_warehouse/events/official_capital_events.json'
URLS=[
 'https://www.csrc.gov.cn/shanxi/c106408/c7555954/content.shtml',
 'https://www.csrc.gov.cn/shanxi/c106408/c7552121/content.shtml',
 'https://www.csrc.gov.cn/anhui/c106363/c7463988/content.shtml',
]
def main():
 rows=[]
 for url in URLS:
  try:
   r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=25); r.raise_for_status()
   if not r.encoding or r.encoding.lower() in ('iso-8859-1','ascii'):
       r.encoding = r.apparent_encoding
   text=re.sub(r'<[^>]+>',' ',r.text)
   text=re.sub(r'\s+',' ',text).strip()
   title='中央汇金/国家队公开事件'
   m=re.search(r'<title[^>]*>(.*?)</title>',r.text,re.I|re.S)
   if m: title=re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',m.group(1))).strip()
   rows.append({'source':'中国证监会','url':url,'title':title,'text':text[:1200],'fetched_at':datetime.now(timezone.utc).isoformat()})
  except Exception as e:
   rows.append({'source':'中国证监会','url':url,'error':str(e)[:180],'fetched_at':datetime.now(timezone.utc).isoformat()})
 OUT.parent.mkdir(parents=True,exist_ok=True); tmp=OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,OUT)
 print(f'official capital events: {len(rows)} -> {OUT}')
if __name__=='__main__': main()
