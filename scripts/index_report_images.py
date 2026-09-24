#!/usr/bin/env python3
"""建立研报图片索引，交给前端视觉模型逐张分析，不做本地OCR。"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated/report_image_index.json'
BASES=[Path.home()/'Desktop'/'研报共享',ROOT/'研究报告',ROOT/'generated']
SUFFIX={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
def main():
 rows=[]; seen=set()
 for base in BASES:
  if not base.exists(): continue
  for p in sorted(base.rglob('*')):
   if p.suffix.lower() not in SUFFIX: continue
   key=str(p.resolve())
   if key in seen: continue
   seen.add(key)
   rows.append({'id':len(rows)+1,'name':p.name,'path':str(p),'folder':str(p.parent),'size':p.stat().st_size})
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'count':len(rows),'items':rows},ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'report image index: {len(rows)}')
if __name__=='__main__': main()
