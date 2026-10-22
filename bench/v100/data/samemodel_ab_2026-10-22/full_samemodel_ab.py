"""Fair same-model A/B: Swift-1.5 Qwen3.8-27B, NVFP4 (1.5 engine) vs Q4_K_M GGUF (1.6 ggufopt engine).
Both routes run the same base model in different quantization, same V100 (device 1), same params,
single lane, greedy. Contexts: 7.6K / 27K / 82K. Modes: plain + MTP d4. Output 96.
Also saves each response for output-divergence comparison."""
import json, subprocess, sys, time
sys.path.insert(0, '/home/max/code/ninfer-v100/bench/v100')
from wait_ready import wait_ready_ip

OUTDIR = "/tmp/samemodel_ab"
subprocess.run(["mkdir","-p",OUTDIR])

NVFP4 = dict(image="ninfer:v1.5.0",
             model="/workspace/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer",
             extra=lambda spec: [])
GGUF  = dict(image="ninfer:1.6.2-ggufopt",
             model="/workspace/models/Swift-1.5-Qwen3.8-27B-Q4_K_M-ninfer-v3.ninfer",
             extra=lambda spec: (["--ngram-draft-tokens","0"] if spec else []))

BASE = ["--host","0.0.0.0","--port","8000","--api-key","sk-g","--model-id","qwen",
        "--max-context","200000","--prefill-chunk","2048","--kv-capacity","auto",
        "--kv-dtype","int8","--max-concurrency","1","--preserve-thinking","--vision"]
MTP = ["--spec","mtp","--draft-tokens","4","--lm-head-draft"]

def filler(n):
    return "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i) for i in range(1, n+1))

CTX = [("7.6K",295), ("27K",1050), ("82K",3267)]

def run(route, route_name, ctx, mode):
    spec = (mode == "mtp")
    name = "ab_%s_%s_%s" % (route_name, ctx[0].replace(".",""), "mtp" if spec else "plain")
    subprocess.run(["docker","rm","-f",name],capture_output=True)
    extra = route["extra"](spec) + (MTP if spec else [])
    subprocess.run(["docker","run","-d","--name",name,"--gpus","device=1",
                    "-v","/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint","ninfer-serve",route["image"],route["model"]]+BASE+extra,
                   capture_output=True)
    log=""
    for _ in range(240):
        p=subprocess.run(["docker","logs",name],capture_output=True,text=True)
        log=p.stdout+p.stderr
        if "listening on" in log or "FATAL" in log: break
        time.sleep(5)
    if "listening on" not in log:
        print("[%s %s %s] 启动失败: %s"%(route_name,ctx[0],mode,log[-400:]),flush=True)
        subprocess.run(["docker","rm","-f",name],capture_output=True); return None
    ip=subprocess.run(["docker","inspect","-f","{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",name],
                      capture_output=True,text=True).stdout.strip()
    wait_ready_ip(ip, timeout_s=300)
    body="/tmp/ab_%s.json"%name
    PROMPT = "秋天到了。" + filler(ctx[1]) + "\n请用三百字写一段关于秋天的散文。"
    json.dump({"model":"qwen","messages":[{"role":"user","content":PROMPT}],
               "max_tokens":96,"temperature":0}, open(body,"w"), ensure_ascii=False)
    outpath = OUTDIR + "/resp_%s_%s_%s.json" % (route_name, ctx[0].replace(".",""), "mtp" if spec else "plain")
    subprocess.run(["curl","-s","--noproxy","*","-m","1800",
                    "http://%s:8000/v1/chat/completions"%ip,"-H","Authorization: Bearer sk-g",
                    "-H","Content-Type: application/json","-d","@"+body,"-o",outpath],capture_output=True)
    p=subprocess.run(["docker","logs",name],capture_output=True,text=True)
    done=[l for l in (p.stdout+p.stderr).splitlines() if "done |" in l][-1:]
    subprocess.run(["docker","rm","-f",name],capture_output=True)
    if not done:
        print("[%s %s %s] 无 done 行"%(route_name,ctx[0],mode),flush=True); return None
    print("[%s %s %s] %s"%(route_name,ctx[0],mode,done[0].strip()),flush=True)
    return done[0].strip()

for route_name, route in [("nvfp4",NVFP4), ("gguf",GGUF)]:
    for ctx in CTX:
        run(route, route_name, ctx, "plain")
        run(route, route_name, ctx, "mtp")
print("=== 完成 ===",flush=True)
