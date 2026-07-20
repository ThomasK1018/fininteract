import json,urllib.request,sys
cfg=json.load(open('configs/openrouter.json',encoding='utf-8'))
req=urllib.request.Request(cfg['base_url'].rstrip('/')+"/credits",
    headers={"Authorization":f"Bearer {cfg['api_key']}"})
d=json.loads(urllib.request.urlopen(req,timeout=20).read())["data"]
print(f"{d['total_usage']:.6f}")
